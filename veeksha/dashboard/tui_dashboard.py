"""Textual-based TUI dashboard for Veeksha benchmarks.

Displays real-time metrics, graphs, and request information with proper log capture.
"""

import logging
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
)
from textual_plotext import PlotextPlot

from veeksha.dashboard.state import DashboardState


class MetricCard(Static):
    """A card displaying a single metric"""

    value = reactive("0")

    def __init__(self, title: str, border_color: str = "blue", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.border_color = border_color

    def compose(self) -> ComposeResult:
        yield Label(self.title, classes="metric-title")
        yield Label(self.value, classes="metric-value")


class PlotextChart(PlotextPlot):
    """Plotext-based line chart for metrics using textual-plotext"""

    data = reactive(list)
    benchmark_start_time = reactive(None)  # Track when benchmark started

    def __init__(
        self, title: str, max_points: int = 100, color: str = "cyan", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.chart_title = title
        self.max_points = max_points
        self.chart_color = color
        self.data = []
        self.benchmark_start_time = None

    def on_mount(self) -> None:
        """Configure the plot when widget is mounted"""
        self.configure_plot()

    def on_resize(self) -> None:
        """Reconfigure plot when widget is resized"""
        self.configure_plot()

    def watch_data(self, new_data: list) -> None:
        """React to data changes and update the plot"""
        self.configure_plot()

    def watch_benchmark_start_time(self, new_time) -> None:
        """React to benchmark start time changes"""
        self.configure_plot()

    def configure_plot(self) -> None:
        """Configure and render the plot"""
        if not self.data or len(self.data) == 0:
            return

        # Get recent data points
        recent_data = list(self.data)[-self.max_points:]

        if len(recent_data) < 2:
            return

        max_val = max(recent_data)
        min_val = min(recent_data)
        avg_val = sum(recent_data) / len(recent_data)

        # Clear and configure the plot
        self.plt.clear_data()
        self.plt.clear_figure()

        # Set cleaner theme with less contrast
        self.plt.theme("clear")
        self.plt.canvas_color("black")
        self.plt.axes_color("black")
        self.plt.ticks_color("gray")

        # Set title
        self.plt.title(self.chart_title)

        # Calculate time-based X-axis
        import time
        if self.benchmark_start_time:
            # Calculate elapsed time for each sample
            current_time = time.time()
            total_elapsed = current_time - self.benchmark_start_time
            # Distribute samples evenly across the elapsed time
            if total_elapsed > 0 and len(recent_data) > 1:
                time_per_sample = total_elapsed / len(recent_data)
                x_vals = [i * time_per_sample for i in range(len(recent_data))]
            else:
                x_vals = list(range(len(recent_data)))
        else:
            # Fallback to sample indices
            x_vals = list(range(len(recent_data)))

        self.plt.plot(
            x_vals, recent_data, color=self.chart_color, marker="braille"
        )

        # Set fixed Y-axis bounds based on min/max with some padding
        y_range = max_val - min_val
        y_padding = y_range * 0.1 if y_range > 0 else 1
        self.plt.ylim(min_val - y_padding, max_val + y_padding)

        # X-axis label shows time and stats
        if self.benchmark_start_time:
            elapsed = time.time() - self.benchmark_start_time
            self.plt.xlabel(f"Time: {elapsed:.1f}s | Avg: {avg_val:.1f} | Min: {min_val:.1f} | Max: {max_val:.1f}")
        else:
            self.plt.xlabel(f"Avg: {avg_val:.1f} | Min: {min_val:.1f} | Max: {max_val:.1f} | Samples: {len(recent_data)}")

        # Minimal grid for cleaner look
        self.plt.grid(False, False)


class LogCapture(logging.Handler):
    """Custom logging handler that captures logs to print after dashboard exits"""

    def __init__(self):
        super().__init__()
        self.buffer = []  # Keep all log entries

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Skip dashboard events to avoid clutter
            if hasattr(record, "dashboard_event"):
                return

            # Add to buffer for later display
            self.buffer.append(msg)
        except Exception:
            # Silently ignore errors
            pass

    def print_logs(self):
        """Print all captured logs to stdout"""
        if self.buffer:
            print("\n" + "=" * 80)
            print("BENCHMARK LOGS")
            print("=" * 80)
            for msg in self.buffer:
                print(msg)
            print("=" * 80 + "\n")


class VeekshaDashboard(App):
    """Textual TUI dashboard for Veeksha benchmarks"""

    CSS = """
    Screen {
        background: $surface;
    }

    .metric-card {
        height: 4;
        border: solid $primary;
        padding: 0;
        margin: 0 1;
    }

    .metric-title {
        text-align: center;
        text-style: bold;
        color: $text-muted;
    }

    .metric-value {
        text-align: center;
        text-style: bold;
        color: $primary;
        height: 2;
        content-align: center middle;
    }

    .chart {
        height: 1fr;
        min-height: 15;
        border: solid $accent;
        padding: 0;
        margin: 0 1;
    }

    .chart-row {
        height: 1fr;
        min-height: 15;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        padding: 1 2;
        margin: 1 0;
        background: $panel;
    }

    #live-requests {
        height: 20;
        border: solid $success;
        margin: 0 1 1 1;
    }

    #completed-requests {
        height: 20;
        border: solid $warning;
        margin: 0 1 1 1;
    }

    .benchmark-selector {
        height: 2;
        border: none;
        padding: 0 1;
        margin: 0;
    }
    
    .metric-row {
        height: 4;
        margin: 0;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("m", "focus_metrics", "Metrics Tab"),
        ("r", "focus_requests", "Requests Tab"),
        ("c", "focus_capacity", "Capacity Search Tab"),
        ("n", "next_benchmark", "Next Benchmark"),
        ("p", "prev_benchmark", "Previous Benchmark"),
    ]

    def __init__(self, dashboard_state: DashboardState):
        super().__init__()
        self.dashboard_state = dashboard_state
        self.update_interval = 1.0  # Update every second
        self.log_handler: Optional[LogCapture] = None

        # Metric cards for Metrics tab
        self.total_requests_card = MetricCard("Total Requests", "blue")
        self.completed_card = MetricCard("Completed", "green")
        self.errors_card = MetricCard("Errors", "red")
        self.duration_card = MetricCard("Duration", "yellow")
        self.ttft_card = MetricCard("Avg TTFT (ms)", "cyan")
        self.tpot_card = MetricCard("Avg TPOT (ms)", "green")
        self.tbt_card = MetricCard("Avg TBT (ms)", "yellow")
        self.latency_card = MetricCard("Avg Latency (ms)", "magenta")

        # Metric cards for Requests tab
        self.total_requests_card_requests = MetricCard("Total Requests", "blue")
        self.completed_card_requests = MetricCard("Completed", "green")
        self.errors_card_requests = MetricCard("Errors", "red")
        self.duration_card_requests = MetricCard("Duration", "yellow")

        # Charts
        self.ttft_chart = PlotextChart("📈 Time to First Token (TTFT)", color="cyan")
        self.tpot_chart = PlotextChart("📉 Time per Output Token (TPOT)", color="green")
        self.tbt_chart = PlotextChart("⏱️  Time Between Tokens (TBT)", color="orange")
        self.latency_chart = PlotextChart("📊 End-to-End Latency", color="magenta")

        # Tables
        self.live_table: Optional[DataTable] = None
        self.completed_table: Optional[DataTable] = None

        # Log capture handler
        self.log_handler: Optional[LogCapture] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with TabbedContent(initial="requests-tab"):
            with TabPane("📋 Requests", id="requests-tab"):
                with ScrollableContainer():
                    # Benchmark selector (also on Requests tab)
                    yield Static(
                        "🎯 Active Benchmark: [bold cyan]Loading...[/bold cyan]",
                        classes="benchmark-selector",
                        id="benchmark-selector-requests",
                    )

                    # Status row - essential info for requests tab
                    with Horizontal(classes="metric-row"):
                        yield self.total_requests_card_requests
                        yield self.completed_card_requests
                        yield self.errors_card_requests
                        yield self.duration_card_requests

                    # Live Requests section
                    yield Static(
                        "🔴 Live Requests (In Progress)", classes="section-title"
                    )
                    self.live_table = DataTable(id="live-requests")
                    self.live_table.add_column("Request ID", key="id")
                    self.live_table.add_column("Input Tokens", key="input")
                    self.live_table.add_column("Output Tokens", key="output")
                    self.live_table.add_column("TTFT (ms)", key="ttft")
                    self.live_table.add_column("TPOT (ms)", key="tpot")
                    self.live_table.add_column("Progress", key="progress")
                    yield self.live_table

                    # Completed Requests section
                    yield Static("✅ Completed Requests", classes="section-title")
                    self.completed_table = DataTable(id="completed-requests")
                    self.completed_table.add_column("Request ID", key="id")
                    self.completed_table.add_column("Input Tokens", key="input")
                    self.completed_table.add_column("Output Tokens", key="output")
                    self.completed_table.add_column("TTFT (ms)", key="ttft")
                    self.completed_table.add_column("TPOT (ms)", key="tpot")
                    yield self.completed_table

            with TabPane("📊 Metrics", id="metrics-tab"):
                with ScrollableContainer():
                    # Benchmark selector
                    yield Static(
                        "🎯 Active Benchmark: [bold cyan]Loading...[/bold cyan]",
                        classes="benchmark-selector",
                        id="benchmark-selector-metrics",
                    )

                    # Compact metric row - only essential metrics
                    with Horizontal(classes="metric-row"):
                        yield self.total_requests_card
                        yield self.ttft_card
                        yield self.completed_card
                        yield self.errors_card
                        yield self.duration_card

                    # Charts in 2x2 grid
                    with Horizontal(classes="chart-row"):
                        yield self.ttft_chart.add_class("chart")
                        yield self.tpot_chart.add_class("chart")

                    with Horizontal(classes="chart-row"):
                        yield self.tbt_chart.add_class("chart")
                        yield self.latency_chart.add_class("chart")

            with TabPane("🔍 Capacity Search", id="capacity-search-tab"):
                with ScrollableContainer():
                    # Status section
                    yield Static("🔍 Capacity Search Status", classes="section-title")
                    yield Static(
                        "Status: [dim]Inactive[/dim]",
                        classes="capacity-status",
                        id="capacity-status",
                    )
                    yield Static(
                        "", classes="capacity-progress", id="capacity-progress"
                    )
                    yield Static("", classes="capacity-range", id="capacity-range")
                    yield Static("", classes="capacity-best", id="capacity-best")

                    # History table
                    yield Static("📊 Test History", classes="section-title")
                    self.capacity_history_table = DataTable(id="capacity-history")
                    self.capacity_history_table.add_column("QPS", key="qps")
                    self.capacity_history_table.add_column("Status", key="status")
                    self.capacity_history_table.add_column("Source", key="source")
                    self.capacity_history_table.add_column(
                        "SLO Metrics", key="slo_metrics"
                    )
                    yield self.capacity_history_table

        yield Footer()

    def on_mount(self) -> None:
        """Set up log capture and start update timer"""
        # Set up log capture
        self.log_handler = LogCapture()
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        # Set log level to INFO to avoid overwhelming the system
        self.log_handler.setLevel(logging.INFO)

        # Add handler to veeksha logger directly (has propagate=False)
        veeksha_logger = logging.getLogger("veeksha")
        veeksha_logger.addHandler(self.log_handler)

        # Start update timer
        self.set_interval(self.update_interval, self.update_dashboard)

    def update_dashboard(self) -> None:
        """Update all dashboard elements"""
        active_id = self.dashboard_state.active_benchmark_id

        # Update both benchmark selectors with running/finished status
        benchmarks = self.dashboard_state.get_benchmark_ids()
        if benchmarks:
            # Determine if benchmark is running or finished
            active_benchmark = self.dashboard_state.get_active_benchmark()
            if active_benchmark:
                is_finished = active_benchmark.benchmark_end_time is not None
                status_indicator = "[bold green]✓ Finished[/bold green]" if is_finished else "[bold yellow]⚙ Running[/bold yellow]"
            else:
                status_indicator = "[dim]No benchmark[/dim]"

            selector_text = (
                f"🎯 Active Benchmark: [bold cyan]{active_id}[/bold cyan] | "
                f"Status: {status_indicator} | "
                f"Press [bold]n[/bold]/[bold]p[/bold] to switch ({len(benchmarks)} total)"
            )

            # Update requests tab selector
            try:
                selector_requests = self.query_one("#benchmark-selector-requests", Static)
                selector_requests.update(selector_text)
            except:
                pass

            # Update metrics tab selector
            try:
                selector_metrics = self.query_one("#benchmark-selector-metrics", Static)
                selector_metrics.update(selector_text)
            except:
                pass

        # Get stats
        stats = self.dashboard_state.get_aggregate_stats(active_id)
        duration = self.dashboard_state.get_benchmark_duration(active_id)

        # Update metric cards on Metrics tab
        self.total_requests_card.value = str(stats.total_requests)
        self.completed_card.value = str(stats.completed_count)
        self.errors_card.value = str(stats.error_count)
        self.duration_card.value = f"{duration:.1f}s"
        self.ttft_card.value = f"{stats.avg_ttft_ms:.1f}ms"
        self.tpot_card.value = f"{stats.avg_tpot_ms:.1f}ms"
        self.tbt_card.value = f"{stats.avg_tbt_ms:.1f}ms"
        self.latency_card.value = f"{stats.avg_latency_ms:.0f}ms"

        # Update metric cards on Requests tab
        self.total_requests_card_requests.value = str(stats.total_requests)
        self.completed_card_requests.value = str(stats.completed_count)
        self.errors_card_requests.value = str(stats.error_count)
        self.duration_card_requests.value = f"{duration:.1f}s"

        # Update charts - directly set data from deques
        self.ttft_chart.data = list(stats.recent_ttft_ms)
        self.tpot_chart.data = list(stats.recent_tpot_ms)
        self.tbt_chart.data = list(stats.recent_tbt_ms)
        self.latency_chart.data = list(stats.recent_latency_ms)

        # Update benchmark start time for time-based X-axis
        active_benchmark = self.dashboard_state.get_active_benchmark()
        if active_benchmark and active_benchmark.benchmark_start_time:
            self.ttft_chart.benchmark_start_time = active_benchmark.benchmark_start_time
            self.tpot_chart.benchmark_start_time = active_benchmark.benchmark_start_time
            self.tbt_chart.benchmark_start_time = active_benchmark.benchmark_start_time
            self.latency_chart.benchmark_start_time = active_benchmark.benchmark_start_time

        # Update live requests table
        if self.live_table:
            self.live_table.clear()
            live_requests = self.dashboard_state.get_live_requests(active_id)
            for req in live_requests[:10]:  # Top 10
                # Create htop-like progress bar
                progress_pct = req.progress_pct if req.progress_pct else 0
                bar_width = 20
                filled = int((progress_pct / 100) * bar_width)
                empty = bar_width - filled
                progress_bar = f"[{'█' * filled}{'░' * empty}] {progress_pct:.0f}%"

                self.live_table.add_row(
                    str(req.request_id),
                    str(req.input_tokens),
                    str(req.current_output_tokens),
                    f"{req.ttft_ms:.1f}" if req.ttft_ms else "-",
                    f"{req.current_tpot_ms:.1f}" if req.current_tpot_ms else "-",
                    progress_bar,
                )

        # Update completed requests table
        if self.completed_table:
            self.completed_table.clear()
            completed = self.dashboard_state.get_completed_requests(active_id)
            for req in list(completed)[-100:]:  # Last 100
                self.completed_table.add_row(
                    str(req.request_id),
                    str(req.input_tokens),
                    str(req.current_output_tokens),
                    f"{req.ttft_ms:.1f}" if req.ttft_ms else "-",
                    f"{req.current_tpot_ms:.1f}" if req.current_tpot_ms else "-",
                )

        # Update capacity search tab
        cs_state = self.dashboard_state.capacity_search
        if cs_state.is_active:
            # Update status
            status_widget = self.query_one("#capacity-status", Static)
            if cs_state.is_complete:
                status_widget.update("Status: [bold green]✓ Complete[/bold green]")
            else:
                status_widget.update(f"Status: [bold yellow]⚙ Running[/bold yellow]")

            # Update progress
            progress_widget = self.query_one("#capacity-progress", Static)
            cache_indicator = (
                " [dim](📦 cached)[/dim]" if cs_state.current_from_cache else ""
            )
            progress_widget.update(
                f"Progress: Iteration {cs_state.current_iteration}/{cs_state.total_iterations} | "
                f"Testing QPS: [bold]{cs_state.current_qps:.1f}[/bold]{cache_indicator}"
            )

            # Update search range
            range_widget = self.query_one("#capacity-range", Static)
            range_widget.update(
                f"Search Range: [{cs_state.search_left:.1f} - {cs_state.search_right:.1f}]"
            )

            # Update best result
            best_widget = self.query_one("#capacity-best", Static)
            if cs_state.best_qps is not None:
                best_widget.update(
                    f"Best QPS: [bold green]{cs_state.best_qps:.1f}[/bold green] ✓"
                )
            else:
                best_widget.update("Best QPS: [dim]Not found yet[/dim]")

            # Update history table
            if self.capacity_history_table:
                self.capacity_history_table.clear()
                for entry in cs_state.qps_history:
                    status_icon = "✓" if entry["under_sla"] else "✗"
                    status_color = "green" if entry["under_sla"] else "red"
                    source = "📦 Cache" if entry.get("from_cache", False) else "🔧 Run"
                    slo_metrics_str = ", ".join(
                        f"{k}={v:.2f}" for k, v in entry["slo_metrics"].items()
                    )

                    self.capacity_history_table.add_row(
                        f"{entry['qps']:.1f}",
                        f"[{status_color}]{status_icon} {'Pass' if entry['under_sla'] else 'Fail'}[/{status_color}]",
                        source,
                        slo_metrics_str or "-",
                    )

    def action_focus_metrics(self) -> None:
        """Switch to metrics tab"""
        tabbed = self.query_one(TabbedContent)
        tabbed.active = "metrics-tab"

    def action_focus_requests(self) -> None:
        """Switch to requests tab"""
        tabbed = self.query_one(TabbedContent)
        tabbed.active = "requests-tab"

    def action_focus_capacity(self) -> None:
        """Switch to capacity search tab"""
        tabbed = self.query_one(TabbedContent)
        tabbed.active = "capacity-search-tab"

    def action_next_benchmark(self) -> None:
        """Switch to next benchmark"""
        benchmarks = self.dashboard_state.get_benchmark_ids()
        if len(benchmarks) <= 1:
            return

        current_idx = benchmarks.index(self.dashboard_state.active_benchmark_id)
        next_idx = (current_idx + 1) % len(benchmarks)
        self.dashboard_state.set_active_benchmark(benchmarks[next_idx])

        # Reset charts when switching benchmarks
        self._reset_charts()

    def action_prev_benchmark(self) -> None:
        """Switch to previous benchmark"""
        benchmarks = self.dashboard_state.get_benchmark_ids()
        if len(benchmarks) <= 1:
            return

        current_idx = benchmarks.index(self.dashboard_state.active_benchmark_id)
        prev_idx = (current_idx - 1) % len(benchmarks)
        self.dashboard_state.set_active_benchmark(benchmarks[prev_idx])

        # Reset charts when switching benchmarks
        self._reset_charts()

    def _reset_charts(self) -> None:
        """Reset all charts (called when switching benchmarks)"""
        self.ttft_chart.data = []
        self.tpot_chart.data = []
        self.tbt_chart.data = []
        self.latency_chart.data = []

        # Reset benchmark start times for new benchmark
        self.ttft_chart.benchmark_start_time = None
        self.tpot_chart.benchmark_start_time = None
        self.tbt_chart.benchmark_start_time = None
        self.latency_chart.benchmark_start_time = None

    def on_unmount(self) -> None:
        """Clean up log handler"""
        if self.log_handler:
            veeksha_logger = logging.getLogger("veeksha")
            veeksha_logger.removeHandler(self.log_handler)


def run_dashboard_tui(dashboard_state: DashboardState) -> None:
    """Run the Textual TUI dashboard in the main thread (blocking).

    Args:
        dashboard_state: The shared dashboard state object

    Note:
        This function blocks until the TUI is closed. It must be called
        from the main thread as Textual requires signal handler registration.
        Logs are captured during execution and printed after the dashboard exits.
    """
    app = VeekshaDashboard(dashboard_state)
    app.run()

    # Print captured logs after dashboard exits
    if app.log_handler:
        app.log_handler.print_logs()
