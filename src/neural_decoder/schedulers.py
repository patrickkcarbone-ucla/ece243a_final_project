"""
Flexible phased schedulers for learning rate and loss weights.

Supports configurable schedules defined as a list of phases:

lr_schedule:
  - name: warmup
    start_epoch: 0
    end_epoch: 5
    function: linear
    start_value: 0.0
    end_value: 0.001
  - name: hold
    start_epoch: 5
    end_epoch: 60
    function: constant
    value: 0.001
  - name: decay
    start_epoch: 60
    end_epoch: 100
    function: cosine
    start_value: 0.001
    end_value: 0.0001

loss_schedule:
  alpha_di:
    - name: ramp
      start_epoch: 0
      end_epoch: 30
      function: linear
      start_value: 0.0
      end_value: 0.4
    - name: hold
      start_epoch: 30
      end_epoch: 100
      function: constant
      value: 0.4
  alpha_mono:
    - name: constant
      start_epoch: 0
      end_epoch: 100
      function: constant
      value: 0.3

Supported functions:
  - linear: Linear interpolation from start_value to end_value
  - constant: Fixed value
  - cosine: Cosine annealing from start_value to end_value
  - exponential: Exponential decay with decay_rate
  - step: Step decay (multiply by decay_rate at each step_size interval)
"""

import math
from typing import List, Dict, Any, Optional, Union


class PhasedValueScheduler:
    """
    Generic epoch-based scheduler for any value (LR, loss weights, etc.).

    Each phase defines:
      - start_epoch, end_epoch: When this phase is active
      - function: How value changes within the phase
      - Value parameters depending on function type

    Supports both old LR-style params (start_lr, end_lr, lr) and new generic
    params (start_value, end_value, value) for backwards compatibility.
    """

    def __init__(
        self, phases: List[Dict[str, Any]], default_value: float = 0.0
    ):
        """
        Args:
            phases: List of phase configs (from YAML)
            default_value: Value to use if no phase covers an epoch
        """
        self.phases = self._validate_and_sort_phases(phases)
        self.default_value = default_value
        self._last_value = None

    def _validate_and_sort_phases(
        self, phases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate phase configs and sort by start_epoch."""
        if not phases:
            raise ValueError("Schedule must have at least one phase")

        sorted_phases = sorted(phases, key=lambda p: p.get("start_epoch", 0))

        for i, phase in enumerate(sorted_phases):
            # Required fields
            if "start_epoch" not in phase:
                raise ValueError(f"Phase {i} missing 'start_epoch'")
            if "end_epoch" not in phase:
                raise ValueError(f"Phase {i} missing 'end_epoch'")
            if "function" not in phase:
                raise ValueError(f"Phase {i} missing 'function'")

            func = phase["function"]

            # Validate function-specific parameters (support both old and new names)
            if func == "linear":
                has_new = "start_value" in phase and "end_value" in phase
                has_old = "start_lr" in phase and "end_lr" in phase
                if not has_new and not has_old:
                    raise ValueError(
                        f"Phase {i} (linear) requires 'start_value'/'end_value' or 'start_lr'/'end_lr'"
                    )
            elif func == "constant":
                if "value" not in phase and "lr" not in phase:
                    raise ValueError(
                        f"Phase {i} (constant) requires 'value' or 'lr'"
                    )
            elif func == "cosine":
                has_new = "start_value" in phase and "end_value" in phase
                has_old = "start_lr" in phase and "end_lr" in phase
                if not has_new and not has_old:
                    raise ValueError(
                        f"Phase {i} (cosine) requires 'start_value'/'end_value' or 'start_lr'/'end_lr'"
                    )
            elif func == "exponential":
                has_start = "start_value" in phase or "start_lr" in phase
                if not has_start or "decay_rate" not in phase:
                    raise ValueError(
                        f"Phase {i} (exponential) requires 'start_value'/'start_lr' and 'decay_rate'"
                    )
            elif func == "step":
                has_start = "start_value" in phase or "start_lr" in phase
                if (
                    not has_start
                    or "decay_rate" not in phase
                    or "step_size" not in phase
                ):
                    raise ValueError(
                        f"Phase {i} (step) requires 'start_value'/'start_lr', 'decay_rate', and 'step_size'"
                    )
            else:
                raise ValueError(f"Unknown function '{func}' in phase {i}")

        return sorted_phases

    def _get_param(
        self, phase: Dict[str, Any], new_name: str, old_name: str
    ) -> float:
        """Get parameter value, supporting both new and old naming conventions."""
        return phase.get(new_name, phase.get(old_name, self.default_value))

    def get_value(self, epoch: int) -> float:
        """
        Get value for a given epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Value for this epoch
        """
        # Find the active phase
        active_phase = None
        prev_phase = None
        for phase in self.phases:
            if phase["start_epoch"] <= epoch < phase["end_epoch"]:
                active_phase = phase
                break
            if phase["end_epoch"] <= epoch:
                prev_phase = phase

        # Handle gaps: use previous phase's end value
        if active_phase is None:
            if prev_phase is not None:
                # In a gap - use previous phase's end value
                func = prev_phase["function"]
                if func == "constant":
                    return self._get_param(prev_phase, "value", "lr")
                elif func in ["linear", "cosine"]:
                    return self._get_param(prev_phase, "end_value", "end_lr")
                elif func in ["exponential", "step"]:
                    return self._compute_phase_value(
                        prev_phase, prev_phase["end_epoch"] - 1
                    )
            # Past all phases or before first phase
            last_phase = self.phases[-1]
            func = last_phase["function"]
            if func == "constant":
                return self._get_param(last_phase, "value", "lr")
            elif func in ["linear", "cosine"]:
                return self._get_param(last_phase, "end_value", "end_lr")
            elif func in ["exponential", "step"]:
                return self._compute_phase_value(
                    last_phase, last_phase["end_epoch"] - 1
                )
            return self.default_value

        return self._compute_phase_value(active_phase, epoch)

    def _compute_phase_value(self, phase: Dict[str, Any], epoch: int) -> float:
        """Compute value within a phase based on its function."""
        func = phase["function"]
        start = phase["start_epoch"]
        end = phase["end_epoch"]

        # Progress within phase (0.0 to 1.0)
        if end > start:
            progress = (epoch - start) / (end - start)
        else:
            progress = 1.0

        if func == "linear":
            start_val = self._get_param(phase, "start_value", "start_lr")
            end_val = self._get_param(phase, "end_value", "end_lr")
            return start_val + progress * (end_val - start_val)

        elif func == "constant":
            return self._get_param(phase, "value", "lr")

        elif func == "cosine":
            start_val = self._get_param(phase, "start_value", "start_lr")
            end_val = self._get_param(phase, "end_value", "end_lr")
            # Cosine annealing: smooth transition
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return end_val + (start_val - end_val) * cosine_factor

        elif func == "exponential":
            start_val = self._get_param(phase, "start_value", "start_lr")
            decay_rate = phase["decay_rate"]
            epochs_in = epoch - start
            return start_val * (decay_rate**epochs_in)

        elif func == "step":
            start_val = self._get_param(phase, "start_value", "start_lr")
            decay_rate = phase["decay_rate"]
            step_size = phase["step_size"]
            epochs_in = epoch - start
            num_steps = epochs_in // step_size
            return start_val * (decay_rate**num_steps)

        return self.default_value

    def step(self, epoch: int) -> float:
        """Get value and store as last value."""
        value = self.get_value(epoch)
        self._last_value = value
        return value


class PhasedLRScheduler(PhasedValueScheduler):
    """
    Epoch-based learning rate scheduler with configurable phases.

    Extends PhasedValueScheduler with optimizer integration.
    """

    def __init__(self, phases: List[Dict[str, Any]], optimizer=None):
        """
        Args:
            phases: List of phase configs (from YAML)
            optimizer: Optional optimizer to update (if None, just computes LR)
        """
        super().__init__(phases, default_value=0.001)
        self.optimizer = optimizer

    def get_lr(self, epoch: int) -> float:
        """Alias for get_value() for backwards compatibility."""
        return self.get_value(epoch)

    def step(self, epoch: int) -> float:
        """
        Update optimizer LR and return new LR.

        Args:
            epoch: Current epoch

        Returns:
            New learning rate
        """
        lr = self.get_value(epoch)
        self._last_value = lr

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        return lr

    def get_last_lr(self) -> List[float]:
        """Return last computed LR (for compatibility with PyTorch schedulers)."""
        if self._last_value is None:
            return [self.get_value(0)]
        return [self._last_value]


class LossScheduler:
    """
    Multi-parameter loss weight scheduler.

    Manages multiple PhasedValueSchedulers for different loss parameters
    (alpha_di, alpha_mono, alpha_context, lambda_gate, etc.)

    Config format:

    loss_schedule:
      alpha_di:
        - name: ramp
          start_epoch: 0
          end_epoch: 30
          function: linear
          start_value: 0.0
          end_value: 0.4
        - name: hold
          start_epoch: 30
          end_epoch: 100
          function: constant
          value: 0.4
      alpha_mono:
        - name: constant
          start_epoch: 0
          end_epoch: 100
          function: constant
          value: 0.3
    """

    def __init__(
        self,
        schedule_config: Dict[str, List[Dict[str, Any]]],
        defaults: Dict[str, float] = None,
    ):
        """
        Args:
            schedule_config: Dict mapping param names to list of phases
            defaults: Default values for each parameter
        """
        self.defaults = defaults or {}
        self.schedulers: Dict[str, PhasedValueScheduler] = {}

        for param_name, phases in schedule_config.items():
            default = self.defaults.get(param_name, 0.0)
            self.schedulers[param_name] = PhasedValueScheduler(
                phases, default_value=default
            )

    def get_values(self, epoch: int) -> Dict[str, float]:
        """
        Get all scheduled values for a given epoch.

        Args:
            epoch: Current epoch

        Returns:
            Dict mapping param names to values
        """
        return {
            name: sched.get_value(epoch)
            for name, sched in self.schedulers.items()
        }

    def get_value(self, param_name: str, epoch: int) -> float:
        """
        Get a specific parameter's value for a given epoch.

        Args:
            param_name: Name of the parameter
            epoch: Current epoch

        Returns:
            Value for this parameter at this epoch
        """
        if param_name in self.schedulers:
            return self.schedulers[param_name].get_value(epoch)
        return self.defaults.get(param_name, 0.0)

    def step(self, epoch: int) -> Dict[str, float]:
        """Step all schedulers and return all values."""
        return {
            name: sched.step(epoch) for name, sched in self.schedulers.items()
        }

    @property
    def param_names(self) -> List[str]:
        """Return list of scheduled parameter names."""
        return list(self.schedulers.keys())


def create_phased_scheduler(cfg, optimizer) -> Optional[PhasedLRScheduler]:
    """
    Create a PhasedLRScheduler from config if lr_schedule is defined.

    Args:
        cfg: Hydra config with optional lr_schedule
        optimizer: PyTorch optimizer

    Returns:
        PhasedLRScheduler if lr_schedule defined, else None
    """
    lr_schedule = getattr(cfg, "lr_schedule", None)

    if lr_schedule is None:
        return None

    # Convert OmegaConf to list of dicts if needed
    from omegaconf import OmegaConf

    if hasattr(lr_schedule, "_iter_ex"):  # OmegaConf ListConfig
        phases = OmegaConf.to_container(lr_schedule, resolve=True)
    else:
        phases = list(lr_schedule)

    return PhasedLRScheduler(phases, optimizer)


def create_loss_scheduler(
    cfg, defaults: Dict[str, float] = None
) -> Optional[LossScheduler]:
    """
    Create a LossScheduler from config if loss_schedule is defined.

    Args:
        cfg: Hydra config with optional loss_schedule
        defaults: Default values for each loss parameter

    Returns:
        LossScheduler if loss_schedule defined, else None

    Example config:

    loss_schedule:
      alpha_di:
        - start_epoch: 0
          end_epoch: 30
          function: linear
          start_value: 0.0
          end_value: 0.4
        - start_epoch: 30
          end_epoch: 100
          function: constant
          value: 0.4
      alpha_mono:
        - start_epoch: 0
          end_epoch: 100
          function: constant
          value: 0.3
    """
    loss_schedule = getattr(cfg, "loss_schedule", None)

    if loss_schedule is None:
        return None

    # Convert OmegaConf to dict of lists if needed
    from omegaconf import OmegaConf

    if hasattr(loss_schedule, "_iter_ex") or hasattr(loss_schedule, "items"):
        schedule_config = OmegaConf.to_container(loss_schedule, resolve=True)
    else:
        schedule_config = dict(loss_schedule)

    # Use provided defaults or extract from cfg
    if defaults is None:
        defaults = {
            "alpha_di": getattr(cfg, "alpha_di", 0.4),
            "alpha_mono": getattr(cfg, "alpha_mono", 0.3),
            "alpha_context": getattr(cfg, "alpha_context", 0.3),
            "lambda_gate": getattr(cfg, "lambda_gate", 0.01),
        }

    return LossScheduler(schedule_config, defaults)


# Preset schedules for convenience
def warmup_cosine_schedule(
    warmup_epochs: int = 5,
    hold_epochs: int = 55,
    decay_epochs: int = 40,
    max_lr: float = 0.001,
    min_lr: float = 0.0001,
) -> List[Dict[str, Any]]:
    """
    Create a warmup -> hold -> cosine decay schedule.

    Example: warmup_cosine_schedule(5, 55, 40, 0.001, 0.0001)
    Creates: 0-5 warmup, 5-60 hold at 0.001, 60-100 cosine to 0.0001
    """
    return [
        {
            "name": "warmup",
            "start_epoch": 0,
            "end_epoch": warmup_epochs,
            "function": "linear",
            "start_lr": 0.0,
            "end_lr": max_lr,
        },
        {
            "name": "hold",
            "start_epoch": warmup_epochs,
            "end_epoch": warmup_epochs + hold_epochs,
            "function": "constant",
            "lr": max_lr,
        },
        {
            "name": "decay",
            "start_epoch": warmup_epochs + hold_epochs,
            "end_epoch": warmup_epochs + hold_epochs + decay_epochs,
            "function": "cosine",
            "start_lr": max_lr,
            "end_lr": min_lr,
        },
    ]


def warmup_step_schedule(
    warmup_epochs: int = 5,
    total_epochs: int = 100,
    max_lr: float = 0.001,
    decay_epoch: int = 60,
    decay_factor: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Create a warmup -> hold -> step decay schedule (current default).

    Example: warmup_step_schedule(5, 100, 0.001, 60, 0.1)
    Creates: 0-5 warmup, 5-60 hold at 0.001, 60-100 hold at 0.0001
    """
    return [
        {
            "name": "warmup",
            "start_epoch": 0,
            "end_epoch": warmup_epochs,
            "function": "linear",
            "start_lr": 0.0,
            "end_lr": max_lr,
        },
        {
            "name": "hold",
            "start_epoch": warmup_epochs,
            "end_epoch": decay_epoch,
            "function": "constant",
            "lr": max_lr,
        },
        {
            "name": "decayed",
            "start_epoch": decay_epoch,
            "end_epoch": total_epochs,
            "function": "constant",
            "lr": max_lr * decay_factor,
        },
    ]
