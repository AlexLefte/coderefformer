import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- utility functions -------------------- #
@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def space_to_depth(x, scale=4):
    """ Equivalent to tf.space_to_depth()
    """

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output


def backward_warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default in PyTorch version
    #        lower than 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


def get_upsampling_func(scale=4, mode='bicubic'):
    if mode == 'bilinear':
        upsample_func = functools.partial(
            F.interpolate, scale_factor=scale, mode=mode,
            align_corners=False)
    elif mode == 'bicubic':
        upsample_func = BicubicUpsample(scale_factor=scale)
    elif mode is None:
        upsample_func = None
    else:
        raise ValueError('Unrecognized upsampling function: {}'.format(mode))

    return upsample_func


# --------------------- utility classes --------------------- #
class BicubicUpsample(nn.Module):
    """ A bicubic upsampling class with similar behavior to that in TecoGAN-Tensorflow

        Note that it's different from torch.nn.functional.interpolate and
        matlab's imresize in terms of bicubic kernel and sampling scheme

        Theoretically it can support any scale_factor >= 1, but currently only
        scale_factor = 4 is tested

        References:
            The original paper: http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
            https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()
        # calculate weights
        cubic = torch.FloatTensor([
            [0, a, -2 * a, a],
            [1, 0, -(a + 3), a + 2],
            [0, -a, (2 * a + 3), -(a + 2)],
            [0, 0, a, -a]
        ])  # accord to Eq.(6) in the reference paper

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
            for s in [1.0*d/scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer('kernels', torch.stack(kernels))

    def forward(self, input):
        n, c, h, w = input.size()
        s = self.scale_factor

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode='replicate')

        # calculate output (height)
        kernel_h = self.kernels.repeat(c, 1).view(-1, 1, s, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0, groups=c)
        output = output.reshape(
            n, c, s, -1, w + 3).permute(0, 1, 3, 2, 4).reshape(n, c, -1, w + 3)

        # calculate output (width)
        kernel_w = self.kernels.repeat(c, 1).view(-1, 1, 1, s)
        output = F.conv2d(output, kernel_w, stride=1, padding=0, groups=c)
        output = output.reshape(
            n, c, s, h * s, -1).permute(0, 1, 3, 4, 2).reshape(n, c, h * s, -1)

        return output

"""
WarmupTrainer — per-module training schedule manager.
 
Replaces the old fix_modules / warm_up_layers / warm_up_iters mechanism with
a single unified config block that supports three modes per module:
 
  fix     — permanently frozen, never trained (replaces fix_modules in generator)
  freeze  — frozen for `iters` iterations, then unfrozen and trained jointly
  warmup  — model frozen globally, ONLY this module trained for `iters` iters,
             then everything unfrozen and trained jointly
 
Each module entry in the config is independent — different modules can have
different modes and different iteration counts.
 
Config example
--------------
warmup:
  modules:
    quantize:
      mode: fix          # permanently frozen — no iters needed
    generator:
      mode: fix          # permanently frozen
    fuse_convs_dict:
      mode: warmup       # train ONLY this for 5000 iters, then jointly
      iters: 5000
    ft_layers:
      mode: freeze       # freeze this for 3000 iters, train everything else
      iters: 3000
    ctx_builders:
      mode: warmup
      iters: 5000
    id_pool:
      mode: warmup
      iters: 5000
 
Notes
-----
- Multiple 'warmup' modules with the same `iters` are all unfrozen at once.
- Multiple 'warmup' modules with different `iters` unfreeze independently.
- 'fix' modules are never unfrozen regardless of iteration count.
- Module names use dot-notation for nested submodules:
    'reconstruction.conv_out' → model.reconstruction.conv_out
"""
class WarmupTrainer:
    """Per-module training schedule with fix / freeze / warmup modes.
 
    Args:
        model  (nn.Module): The network to manage.
        config (dict):      The 'warmup' section from the YAML config.
                            Must contain a 'modules' dict mapping module
                            names to {'mode': str, 'iters': int} entries.
 
    Attributes:
        schedule (dict): Parsed per-module entries, keyed by module name.
                         Each value: {'mode', 'iters', 'module', 'done'}
    """
 
    VALID_MODES = ('fix', 'freeze', 'warmup')
 
    def __init__(self, model: nn.Module, config: dict):
        self.model    = model
        self.schedule = {}   # name → {mode, iters, module, done}
        self._logger_lines = []
 
        modules_cfg = config.get('modules', {})
        if not modules_cfg:
            return
 
        # --- Parse config and apply initial freezing ---
        for name, entry in modules_cfg.items():
            mode  = entry.get('mode', None)
            iters = entry.get('iters', 0)
 
            if mode not in self.VALID_MODES:
                print(f"[WarmupTrainer] WARNING: unknown mode '{mode}' for "
                      f"module '{name}'. Expected one of {self.VALID_MODES}.")
                continue
 
            module = self._get_module(name)
            if module is None:
                continue
 
            self.schedule[name] = {
                'mode':   mode,
                'iters':  iters,
                'module': module,
                'done':   False,
            }
 
        # Apply initial state: fix and freeze require immediate action.
        # Warmup requires freezing the ENTIRE model first (done after full parse).
        has_warmup = any(e['mode'] == 'warmup' for e in self.schedule.values())
 
        if has_warmup:
            # Freeze entire model — warmup modules will be unfrozen below
            self._set_grad(self.model, False)
            print("[WarmupTrainer] Warmup mode detected — freezing entire model.")
 
        for name, entry in self.schedule.items():
            mode   = entry['mode']
            module = entry['module']
            iters  = entry['iters']
 
            if mode == 'fix':
                # Permanently frozen — requires_grad=False forever
                self._set_grad(module, False)
                entry['done'] = True   # no step() action needed
                print(f"[WarmupTrainer] '{name}' → fix (permanent freeze)")
 
            elif mode == 'freeze':
                # Frozen for `iters` iters; if whole-model freeze was applied
                # above for warmup modules, this is already frozen — no-op.
                # If no warmup modules exist, freeze explicitly.
                if not has_warmup:
                    self._set_grad(module, False)
                print(f"[WarmupTrainer] '{name}' → freeze for {iters} iters")
 
            elif mode == 'warmup':
                # Only this module should be trainable → unfreeze it
                self._set_grad(module, True)
                print(f"[WarmupTrainer] '{name}' → warmup for {iters} iters "
                      f"(only this module trains until iter {iters})")
 
    # ------------------------------------------------------------------
    def step(self, current_iter: int):
        """Advance the schedule. Call once per training iteration.
 
        Checks each non-done module and unfreezes it when its `iters`
        threshold is reached.  'fix' entries are never processed here.
        """
        for name, entry in self.schedule.items():
            if entry['done']:
                continue
 
            mode  = entry['mode']
            iters = entry['iters']
 
            if current_iter < iters:
                continue
 
            # Threshold reached — unfreeze according to mode
            if mode == 'warmup':
                # Unfreeze the entire model, then re-apply all remaining
                # frozen/fix states so we don't accidentally unfreeze them
                print(f"[WarmupTrainer] iter {current_iter}: "
                      f"'{name}' warmup done — unfreezing full model.")
                self._set_grad(self.model, True)
                self._reapply_permanent()
 
            elif mode == 'freeze':
                # Unfreeze only this module
                self._set_grad(entry['module'], True)
                print(f"[WarmupTrainer] iter {current_iter}: "
                      f"'{name}' freeze done — module unfrozen.")
 
            entry['done'] = True
 
    # ------------------------------------------------------------------
    def is_module_frozen(self, name: str) -> bool:
        """Return True if the named module is currently frozen."""
        entry = self.schedule.get(name)
        if entry is None:
            return False
        m = entry['module']
        # Handle bare nn.Parameter — no .parameters() iterator.
        if isinstance(m, nn.Parameter):
            return not m.requires_grad
        return not next(m.parameters()).requires_grad
 
    # ------------------------------------------------------------------
    def _set_grad(self, module, flag: bool):
        # Handle bare nn.Parameter (e.g. backbone.position_emb) — these are
        # direct tensor attributes on the module, not nn.Module submodules,
        # so they have no .parameters() method.
        if isinstance(module, nn.Parameter):
            module.requires_grad = flag
            return
        for p in module.parameters():
            p.requires_grad = flag
 
    def _get_module(self, name: str):
        """Retrieve submodule by dot-notation name or search pattern.
        
        Three modes:
        - Exact path:   'ft_layers'  → direct attribute lookup
        - Suffix match: '*.pcd_align' or just 'pcd_align' → find all
                        submodules whose name ends with 'pcd_align'
        - Substring:    '*pcd*' → find all submodules containing 'pcd'
        
        When multiple modules match, returns a _ModuleGroup.
        """
        print(f"Searching for: {name}..")

        # Exact path — original behaviour
        if '*' not in name:
            module = self.model
            for part in name.split('.'):
                module = getattr(module, part, None)
                if module is None:
                    print(f"[WarmupTrainer] WARNING: '{name}' not found.")
                    return None
            return module

        # Pattern match — search all named submodules
        # Strip leading/trailing '*' to get the search term
        pattern = name.replace('*', '')

        matched = [
            module
            for full_name, module in self.model.named_modules()
            if pattern in full_name and module is not self.model
        ]

        if not matched:
            print(f"[WarmupTrainer] WARNING: no modules matched pattern '{name}'.")
            return None

        print(f"[WarmupTrainer] Pattern '{name}' matched {len(matched)} modules:")
        for full_name, _ in self.model.named_modules():
            if pattern in full_name:
                print(f"  → {full_name}")

        return _ModuleGroup(matched)
 
    def _reapply_permanent(self):
        """After a global unfreeze, re-freeze fix and still-active entries."""
        for name, entry in self.schedule.items():
            mode = entry['mode']
            if mode == 'fix':
                # Always keep fixed modules frozen
                self._set_grad(entry['module'], False)
            elif not entry['done']:
                # freeze/warmup entries whose iters haven't been reached yet
                self._set_grad(entry['module'], False)


class _ModuleGroup:
    """Wraps multiple modules so WarmupTrainer can treat them as one unit."""

    def __init__(self, modules: list):
        self._modules = modules

    def parameters(self):
        for m in self._modules:
            yield from m.parameters()