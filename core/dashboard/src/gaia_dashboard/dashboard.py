import time
import numpy as np
import psutil
from collections import deque
from threading import Thread

import torch
import rich
from rich.table import Table
from rich.console import Console


class Utilization(Thread):
    """Thread that monitors CPU, memory, and GPU utilization."""

    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100 * psutil.cpu_percent() / psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100 * mem.active / mem.total)
            if torch.cuda.is_available():
                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100 * (total - free) / total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


class Dashboard:
    """Simple rich dashboard for training visualization."""

    def __init__(self):
        self.start_time = time.time()
        self.utilization = Utilization()
        self.training_loss = None
        self.validation_loss = None
        self.epoch = 0
        self._idx = 0

    @property
    def uptime(self):
        return time.time() - self.start_time

    def update(self, training_loss=None, validation_loss=None, epoch=None, auto_print=True):
        """Update dashboard metrics and optionally print the dashboard.
        
        Args:
            training_loss: Training loss value to update
            validation_loss: Validation loss value to update
            epoch: Epoch number to update
            auto_print: If True, automatically print the dashboard after updating (default: True)
        """
        if training_loss is not None:
            self.training_loss = training_loss
        if validation_loss is not None:
            self.validation_loss = validation_loss
        if epoch is not None:
            self.epoch = epoch
        
        if auto_print:
            self.print_dashboard()

    def print_dashboard(self, clear=False):
        """Print the dashboard using rich tables."""
        console = Console()
        dashboard = Table(
            box=rich.box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="bright_white",
        )
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f"[bright_white]gaia[/bright_white] 🧠",
            f"[bright_white]CPU:[/bright_white] [default]{np.mean(self.utilization.cpu_util):.1f}[/default][dim]%[/dim]",
            f"[bright_white]GPU:[/bright_white] [default]{np.mean(self.utilization.gpu_util):.1f}[/default][dim]%[/dim]",
            f"[bright_white]DRAM:[/bright_white] [default]{np.mean(self.utilization.cpu_mem):.1f}[/default][dim]%[/dim]",
            f"[bright_white]VRAM:[/bright_white] [default]{np.mean(self.utilization.gpu_mem):.1f}[/default][dim]%[/dim]",
        )

        # Losses table
        losses_table = Table(box=None, expand=True)
        losses_table.add_column(
            "[bright_white]Metric[/bright_white]", justify="left", width=20
        )
        losses_table.add_column(
            "[bright_white]Value[/bright_white]", justify="right", width=12
        )

        if self.training_loss is not None:
            losses_table.add_row(
                "[default]Training Loss[/default]",
                f"[default]{self.training_loss:.6f}[/default]",
            )
        if self.validation_loss is not None:
            losses_table.add_row(
                "[default]Validation Loss[/default]",
                f"[default]{self.validation_loss:.6f}[/default]",
            )

        losses_table.add_row(
            "[default]Epoch[/default]", f"[default]{self.epoch}[/default]"
        )
        losses_table.add_row(
            "[default]Uptime[/default]", self._format_duration(self.uptime)
        )

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(losses_table)
        dashboard.add_row(monitor)

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print("\033[2J\033[H" + capture.get())

    def _format_duration(self, seconds):
        """Format duration as hours, minutes, seconds."""
        if seconds < 0:
            return "[default]0[/default][dim]s[/dim]"
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h:
            return f"[default]{h}[/default][dim]h[/dim] [default]{m}[/default][dim]m[/dim] [default]{s}[/default][dim]s[/dim]"
        elif m:
            return (
                f"[default]{m}[/default][dim]m[/dim] [default]{s}[/default][dim]s[/dim]"
            )
        else:
            return f"[default]{s}[/default][dim]s[/dim]"

    def close(self):
        """Stop the dashboard and cleanup resources."""
        self.utilization.stop()
