from __future__ import annotations

import sys
import itertools
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from qcodes.station import Station
from qcodes.dataset import initialise_or_create_database_at, load_or_create_experiment
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter

import trigger_fns
import utilities


@dataclass
class ChannelConfig:
    channel_name: str
    name: str
    waveform: str
    start_voltage: float
    first_node: float
    second_node: float
    dV: float
    v_high: float
    v_low: float
    v_mid: float
    v_fixed: float
    n_high: int
    n_low: int
    n_mid: int
    n_ramp: int
    n_offset: int
    v_amp: float
    v_offset: float
    n_period: int
    independent: bool
    link_next: bool


class WaveformPlot(FigureCanvasQTAgg):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.fig = Figure(figsize=(7, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(1, 1, 1)

    def plot(self, traces: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.ax.clear()
        for name, (t, v) in traces.items():
            self.ax.plot(t, v, label=name)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        if traces:
            self.ax.legend(loc="best")
        self.draw()


class ArbitrarySweeperGUI(QtWidgets.QMainWindow):
    COL_CHANNEL = 0
    COL_NAME = 1
    COL_WAVEFORM = 2
    COL_INDEP = 3
    COL_LINK = 4

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Arbitrary Sweeper Trigger GUI")

        self.station: Station | None = None
        self.keithleys: dict[str, Any] = {}
        self.run_thread: QtCore.QThread | None = None
        self.run_worker: RunWorker | None = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main = QtWidgets.QVBoxLayout(root)

        conn_block = self._build_connection_block()
        params_block = self._build_params_block()
        options_block = self._build_options_block()
        plot_block = self._build_plot_block()

        main.addWidget(conn_block)

        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom.addWidget(options_block)
        bottom.addWidget(plot_block)
        bottom.setChildrenCollapsible(False)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)
        bottom.setSizes([500, 500])
        bottom.setHandleWidth(6)
        bottom.setStyleSheet("QSplitter::handle{background: #c0c0c0;}")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(params_block)
        splitter.addWidget(bottom)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 500])
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("QSplitter::handle{background: #c0c0c0;}")

        main.addWidget(splitter, 1)

        self._set_defaults()

    def _build_connection_block(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Connect / Paths")
        layout = QtWidgets.QGridLayout(box)

        self.yaml_path = QtWidgets.QLineEdit("electrochemistry.station.sim.yaml")
        self.db_path = QtWidgets.QLineEdit("../_data/db/sim/test.db")
        self.csv_path = QtWidgets.QLineEdit("../_data/csv/sim/test.csv")
        self.exp_name = QtWidgets.QLineEdit("test")
        self.device_name = QtWidgets.QLineEdit("test")

        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        self.connect_status = QtWidgets.QLabel("Disconnected")

        self.make_db_btn = QtWidgets.QPushButton("Make DB")
        self.make_db_btn.clicked.connect(self._on_make_db)
        self.db_status = QtWidgets.QLabel("")

        self.run_indicator = QtWidgets.QLabel()
        self.run_indicator.setFixedSize(12, 12)
        self._set_indicator("idle")
        self.run_status = QtWidgets.QLabel("Idle")

        layout.addWidget(QtWidgets.QLabel("YAML config path"), 0, 0)
        layout.addWidget(self.yaml_path, 0, 1, 1, 3)

        layout.addWidget(QtWidgets.QLabel("DB save path"), 1, 0)
        layout.addWidget(self.db_path, 1, 1, 1, 3)

        layout.addWidget(QtWidgets.QLabel("CSV save path"), 2, 0)
        layout.addWidget(self.csv_path, 2, 1, 1, 3)

        layout.addWidget(QtWidgets.QLabel("Experiment name"), 3, 0)
        layout.addWidget(self.exp_name, 3, 1, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Device name"), 3, 2)
        layout.addWidget(self.device_name, 3, 3)

        layout.addWidget(self.connect_btn, 4, 0)
        layout.addWidget(self.connect_status, 4, 1)
        layout.addWidget(self.make_db_btn, 4, 2)
        layout.addWidget(self.db_status, 4, 3)

        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self.run_indicator)
        status_row.addWidget(self.run_status)
        status_row.addStretch(1)
        layout.addLayout(status_row, 5, 0, 1, 4)

        return box

    def _build_params_block(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Channels / Waveforms")
        layout = QtWidgets.QVBoxLayout(box)

        # Left table: compact channel list.
        self.channel_table = QtWidgets.QTableWidget(0, 5)
        self.channel_table.setHorizontalHeaderLabels(
            [
                "Channel",
                "Name",
                "Waveform",
                "Independent",
                "Link Next",
            ]
        )
        self.channel_table.horizontalHeader().setStretchLastSection(True)
        self.channel_table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.channel_table.selectionModel().currentRowChanged.connect(
            self._on_row_selected
        )

        btn_row = QtWidgets.QHBoxLayout()
        self.move_up_btn = QtWidgets.QPushButton("Move Up")
        self.move_down_btn = QtWidgets.QPushButton("Move Down")
        self.move_up_btn.clicked.connect(self._move_row_up)
        self.move_down_btn.clicked.connect(self._move_row_down)
        btn_row.addWidget(self.move_up_btn)
        btn_row.addWidget(self.move_down_btn)
        btn_row.addStretch(1)

        layout.addLayout(btn_row)

        # Right panel: per-channel detail editor.
        self.detail_panel = self._build_detail_panel()

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.channel_table)
        split.addWidget(self.detail_panel)
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([600, 300])
        split.setHandleWidth(6)
        split.setStyleSheet("QSplitter::handle{background: #c0c0c0;}")

        layout.addWidget(split, 1)

        return box

    def _build_options_block(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Options")
        layout = QtWidgets.QGridLayout(box)

        self.time_independent = QtWidgets.QCheckBox()
        self.ramp_up = QtWidgets.QCheckBox()
        self.ramp_down = QtWidgets.QCheckBox()
        self.temp_measure = QtWidgets.QCheckBox()
        self.dt_list = QtWidgets.QLineEdit("0.5")
        self.delayNPLC_ratio = QtWidgets.QLineEdit("0.8")
        self.repeat = QtWidgets.QLineEdit("1")
        self.round_delay = QtWidgets.QLineEdit("0")

        layout.addWidget(QtWidgets.QLabel("time_independent"), 0, 0)
        layout.addWidget(self.time_independent, 0, 1)
        layout.addWidget(QtWidgets.QLabel("ramp_up"), 0, 2)
        layout.addWidget(self.ramp_up, 0, 3)

        layout.addWidget(QtWidgets.QLabel("ramp_down"), 1, 0)
        layout.addWidget(self.ramp_down, 1, 1)
        layout.addWidget(QtWidgets.QLabel("temp_measure"), 1, 2)
        layout.addWidget(self.temp_measure, 1, 3)

        self.ramp_to_zero_btn = QtWidgets.QPushButton("Ramp Channels To 0")
        self.ramp_to_zero_btn.clicked.connect(self._on_ramp_to_zero)
        layout.addWidget(self.ramp_to_zero_btn, 2, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("dt_list (comma)"), 2, 2)
        layout.addWidget(self.dt_list, 2, 3)

        layout.addWidget(QtWidgets.QLabel("delayNPLC_ratio"), 3, 0)
        layout.addWidget(self.delayNPLC_ratio, 3, 1)
        layout.addWidget(QtWidgets.QLabel("repeat"), 3, 2)
        layout.addWidget(self.repeat, 3, 3)

        layout.addWidget(QtWidgets.QLabel("round_delay (s)"), 4, 0)
        layout.addWidget(self.round_delay, 4, 1)

        self.plot_btn = QtWidgets.QPushButton("Plot Waveforms")
        self.plot_btn.clicked.connect(self._on_plot)
        layout.addWidget(self.plot_btn, 5, 0, 1, 2)

        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        layout.addWidget(self.run_btn, 5, 2, 1, 2)

        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause_resume)
        layout.addWidget(self.pause_btn, 6, 0, 1, 2)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self.stop_btn, 6, 2, 1, 2)

        return box

    def _build_plot_block(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Waveforms vs Time")
        layout = QtWidgets.QVBoxLayout(box)
        self.plot = WaveformPlot()
        layout.addWidget(self.plot)
        return box

    def _set_defaults(self) -> None:
        self.time_independent.setChecked(False)
        self.ramp_up.setChecked(True)
        self.ramp_down.setChecked(False)
        self.temp_measure.setChecked(True)

    def _on_connect(self) -> None:
        yaml_path = self.yaml_path.text().strip()
        if not yaml_path:
            self.connect_status.setText("YAML path required")
            return

        try:
            self.station = Station(config_file=yaml_path)
            instruments = self._load_yaml_instruments(yaml_path)
            self.keithleys.clear()
            for name in instruments:
                if name.startswith("keithley"):
                    loader = getattr(self.station, f"load_{name}", None)
                    if loader is None:
                        continue
                    self.keithleys[name] = loader()

            if not self.keithleys:
                self.connect_status.setText("No keithleys found")
                return

            self.connect_status.setText("Connected")
            self._populate_channels()
        except Exception as exc:
            self.connect_status.setText(f"Connect failed: {exc}")

    def _on_make_db(self) -> None:
        db_path = self.db_path.text().strip()
        if not db_path:
            self.db_status.setText("DB path required")
            return
        try:
            initialise_or_create_database_at(db_path)
            self.db_status.setText("DB ready")
        except Exception as exc:
            self.db_status.setText(f"DB error: {exc}")

    def _populate_channels(self) -> None:
        self.channel_table.setRowCount(0)
        for kname, inst in self.keithleys.items():
            for ch in ["smua", "smub"]:
                self._add_channel_row(f"{kname}.{ch}")

    def _add_channel_row(self, channel_name: str) -> None:
        row = self.channel_table.rowCount()
        self.channel_table.insertRow(row)

        self.channel_table.setItem(
            row, self.COL_CHANNEL, QtWidgets.QTableWidgetItem(channel_name)
        )
        self.channel_table.setItem(
            row, self.COL_NAME, QtWidgets.QTableWidgetItem(channel_name)
        )

        combo = QtWidgets.QComboBox()
        combo.addItems(["Triangle", "Square", "Square-3", "Sine", "Fixed"])
        combo.setCurrentText("Triangle")
        combo.setProperty("row", row)
        combo.currentTextChanged.connect(self._on_waveform_changed_for_widget)
        self.channel_table.setCellWidget(row, self.COL_WAVEFORM, combo)

        indep = QtWidgets.QTableWidgetItem()
        indep.setFlags(indep.flags() | QtCore.Qt.ItemIsUserCheckable)
        indep.setCheckState(QtCore.Qt.Unchecked)
        self.channel_table.setItem(row, self.COL_INDEP, indep)

        link_next = QtWidgets.QTableWidgetItem()
        link_next.setFlags(link_next.flags() | QtCore.Qt.ItemIsUserCheckable)
        link_next.setCheckState(QtCore.Qt.Unchecked)
        self.channel_table.setItem(row, self.COL_LINK, link_next)

        # Initialize per-row detail state.
        self._set_row_state_defaults(row)
        if row == 0:
            self.channel_table.selectRow(0)
            self._load_details_from_row(0)

    def _move_row_up(self) -> None:
        row = self.channel_table.currentRow()
        if row <= 0:
            return
        self._swap_rows(row, row - 1)
        self.channel_table.selectRow(row - 1)

    def _move_row_down(self) -> None:
        row = self.channel_table.currentRow()
        if row < 0 or row >= self.channel_table.rowCount() - 1:
            return
        self._swap_rows(row, row + 1)
        self.channel_table.selectRow(row + 1)

    def _swap_rows(self, a: int, b: int) -> None:
        row_a = self._get_row_data(a)
        row_b = self._get_row_data(b)
        self._set_row_data(a, row_b)
        self._set_row_data(b, row_a)

    def _on_plot(self) -> None:
        try:
            configs = self._collect_channel_configs()
            dt_list = self._parse_float_list(self.dt_list.text())
            repeat = int(self.repeat.text().strip() or "1")
            round_delay = float(self.round_delay.text().strip() or "0")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", str(exc))
            return

        traces = self._build_traces(configs, dt_list, repeat, round_delay)
        self.plot.plot(traces)

    def _collect_channel_configs(self) -> list[ChannelConfig]:
        configs: list[ChannelConfig] = []
        for row in range(self.channel_table.rowCount()):
            channel_name = self.channel_table.item(
                row, self.COL_CHANNEL
            ).text().strip()
            name = self.channel_table.item(row, self.COL_NAME).text().strip()
            waveform = self._get_waveform_value(row)

            state = self._get_row_state(row)
            start_voltage = float(state["start_voltage"])
            first_node = float(state["first_node"])
            second_node = float(state["second_node"])
            dV = float(state["dV"])
            v_high = float(state["v_high"])
            v_low = float(state["v_low"])
            n_high = int(state["n_high"])
            n_low = int(state["n_low"])
            n_ramp = int(state["n_ramp"])
            n_offset = int(state["n_offset"])
            v_mid = float(state["v_mid"])
            v_fixed = float(state["v_fixed"])
            n_mid = int(state["n_mid"])
            v_amp = float(state["v_amp"])
            v_offset = float(state["v_offset"])
            n_period = int(state["n_period"])

            independent = (
                self.channel_table.item(row, self.COL_INDEP).checkState() == QtCore.Qt.Checked
            )
            link_next = (
                self.channel_table.item(row, self.COL_LINK).checkState() == QtCore.Qt.Checked
            )

            configs.append(
                ChannelConfig(
                    channel_name=channel_name,
                    name=name,
                    waveform=waveform,
                    start_voltage=start_voltage,
                    first_node=first_node,
                    second_node=second_node,
                    dV=dV,
                    v_high=v_high,
                    v_low=v_low,
                    v_mid=v_mid,
                    v_fixed=v_fixed,
                    n_high=n_high,
                    n_low=n_low,
                    n_mid=n_mid,
                    n_ramp=n_ramp,
                    n_offset=n_offset,
                    v_amp=v_amp,
                    v_offset=v_offset,
                    n_period=n_period,
                    independent=independent,
                    link_next=link_next,
                )
            )
        return configs

    @staticmethod
    def _parse_float_list(text: str) -> list[float]:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError("dt_list is empty")
        return [float(p) for p in parts]

    @staticmethod
    def _build_v_range(cfg: ChannelConfig) -> np.ndarray:
        if cfg.waveform.lower() == "square":
            return ArbitrarySweeperGUI._build_square_wave(cfg)
        if cfg.waveform.lower() == "square-3":
            return ArbitrarySweeperGUI._build_square3_wave(cfg)
        if cfg.waveform.lower() == "sine":
            return ArbitrarySweeperGUI._build_sine_wave(cfg)
        if cfg.waveform.lower() == "fixed":
            return np.array([cfg.v_fixed], dtype=float)

        if cfg.dV == 0:
            return np.array([cfg.start_voltage], dtype=float)

        n = 1 + abs(int((cfg.first_node - cfg.start_voltage) / cfg.dV))
        v_range1 = np.linspace(cfg.start_voltage, cfg.first_node, n)[:-1]
        n = 1 + abs(int((cfg.second_node - cfg.first_node) / cfg.dV))
        v_range2 = np.linspace(cfg.first_node, cfg.second_node, n)[:-1]
        n = 1 + abs(int((cfg.start_voltage - cfg.second_node) / cfg.dV))
        v_range3 = np.linspace(cfg.second_node, cfg.start_voltage, n)
        return np.concatenate((v_range1, v_range2, v_range3))

    @staticmethod
    def _build_square_wave(cfg: ChannelConfig) -> np.ndarray:
        n_high = max(0, int(cfg.n_high))
        n_low = max(0, int(cfg.n_low))
        n_ramp = max(0, int(cfg.n_ramp))
        v_high = float(cfg.v_high)
        v_low = float(cfg.v_low)
        n_offset = int(cfg.n_offset)

        if n_ramp > 0:
            ramp_up = np.linspace(v_low, v_high, n_ramp + 2)[1:-1]
            ramp_down = np.linspace(v_high, v_low, n_ramp + 2)[1:-1]
        else:
            ramp_up = np.array([], dtype=float)
            ramp_down = np.array([], dtype=float)

        high = np.full(n_high, v_high, dtype=float)
        low = np.full(n_low, v_low, dtype=float)

        cycle = np.concatenate((low, ramp_up, high, ramp_down))
        if cycle.size == 0:
            return np.array([v_low], dtype=float)

        shift = n_offset % cycle.size
        if shift:
            cycle = np.concatenate((cycle[shift:], cycle[:shift]))
        return cycle

    @staticmethod
    def _build_square3_wave(cfg: ChannelConfig) -> np.ndarray:
        v_high = float(cfg.v_high)
        v_low = float(cfg.v_low)
        v_mid = float(cfg.v_mid)
        n_high = max(0, int(cfg.n_high))
        n_low = max(0, int(cfg.n_low))
        n_mid = max(0, int(cfg.n_mid))
        n_offset = int(cfg.n_offset)

        mid = np.full(n_mid, v_mid, dtype=float)
        low = np.full(n_low, v_low, dtype=float)
        high = np.full(n_high, v_high, dtype=float)

        cycle = np.concatenate((mid, low, mid, high))
        if cycle.size == 0:
            return np.array([v_mid], dtype=float)

        shift = n_offset % cycle.size
        if shift:
            cycle = np.concatenate((cycle[shift:], cycle[:shift]))
        return cycle

    @staticmethod
    def _build_sine_wave(cfg: ChannelConfig) -> np.ndarray:
        v_amp = float(cfg.v_amp)
        v_offset = float(cfg.v_offset)
        n_period = max(1, int(cfg.n_period))
        t = np.linspace(0, 2 * np.pi, n_period, endpoint=False)
        return v_offset + v_amp * np.sin(t)

    def _build_traces(
        self,
        configs: list[ChannelConfig],
        dt_list: list[float],
        repeat: int,
        round_delay: float,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        v_ranges = [self._build_v_range(cfg) for cfg in configs]
        groups = self._build_groups(configs)

        # Build measurement sequence (list of voltage tuples)
        sequence: list[tuple[float, ...]] = []
        for _dt in dt_list:
            for _rep in range(repeat):
                seq = self._iterate_groups(groups, v_ranges)
                sequence.extend(seq)
                if round_delay > 0:
                    sequence.append(tuple(v[-1] for v in v_ranges))

        dt = dt_list[0]
        t = np.arange(len(sequence)) * dt

        traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for idx, cfg in enumerate(configs):
            traces[cfg.name] = (t, np.array([s[idx] for s in sequence], dtype=float))
        return traces

    @staticmethod
    def _build_groups(configs: list[ChannelConfig]) -> list[list[int]]:
        groups: list[list[int]] = []
        current: list[int] = []
        for idx, cfg in enumerate(configs):
            current.append(idx)
            if not cfg.link_next:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _iterate_groups(
        groups: list[list[int]], v_ranges: list[np.ndarray]
    ) -> list[tuple[float, ...]]:
        # Build per-group iterables: linked groups advance together by index.
        group_iters: list[list[tuple[float, ...]]] = []
        for group in groups:
            group_ranges = [v_ranges[i] for i in group]
            max_len = max(len(r) for r in group_ranges)
            padded = [
                np.pad(r, (0, max_len - len(r)), mode="edge") for r in group_ranges
            ]
            group_iters.append(list(zip(*padded)))

        # Nested loops across groups
        sequence: list[tuple[float, ...]] = []
        for combo in itertools.product(*group_iters):
            # combo is tuple of tuples, one per group
            flat = [0.0] * len(v_ranges)
            for group, values in zip(groups, combo):
                for idx, val in zip(group, values):
                    flat[idx] = val
            sequence.append(tuple(flat))
        return sequence

    def _build_sweepers(
        self, configs: list[ChannelConfig]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sweepers: list[dict[str, Any]] = []
        for cfg in configs:
            inst_name, ch_name = cfg.channel_name.split(".")
            channel = getattr(self.keithleys[inst_name], ch_name)
            v_range = self._build_v_range(cfg)
            sweepers.append(
                {
                    "channel": channel,
                    "name": cfg.name,
                    "first_node": cfg.first_node,
                    "second_node": cfg.second_node,
                    "start_voltage": cfg.start_voltage,
                    "dV": cfg.dV,
                    "independent": cfg.independent,
                    "v_range": v_range,
                }
            )

        # Preserve current order for saving; same as GUI order for now
        return sweepers, list(sweepers)

    @staticmethod
    def _resolve_csv_path(base: str, device: str, exp: str, run_id: int) -> str:
        if base.endswith(".csv"):
            return base
        if base.endswith("\\") or base.endswith("/"):
            return f"{base}{device}{exp}_{run_id}_manual_sweep.csv"
        return f"{base}\\{device}{exp}_{run_id}_manual_sweep.csv"

    def _get_waveform_value(self, row: int) -> str:
        widget = self.channel_table.cellWidget(row, self.COL_WAVEFORM)
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        return "Triangle"

    def _on_waveform_changed_for_widget(self, _value: str) -> None:
        combo = self.sender()
        if not isinstance(combo, QtWidgets.QComboBox):
            return
        row = combo.property("row")
        if row is None:
            return
        if self.channel_table.currentRow() == int(row):
            self._update_detail_visibility(combo.currentText())

    def _get_row_data(self, row: int) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for col in range(self.channel_table.columnCount()):
            widget = self.channel_table.cellWidget(row, col)
            if isinstance(widget, QtWidgets.QComboBox):
                data[col] = ("combo", widget.currentText())
            else:
                item = self.channel_table.item(row, col)
                if item is None:
                    data[col] = ("item", "", QtCore.Qt.Unchecked)
                else:
                    data[col] = ("item", item.text(), item.checkState())
        state_item = self.channel_table.item(row, self.COL_CHANNEL)
        if state_item is not None:
            data["__state__"] = state_item.data(QtCore.Qt.UserRole)
        return data

    def _set_row_data(self, row: int, data: dict[str, Any]) -> None:
        for col, value in data.items():
            if col == "__state__":
                continue
            if value[0] == "combo":
                combo = QtWidgets.QComboBox()
                combo.addItems(["Triangle", "Square", "Square-3", "Sine", "Fixed"])
                combo.setCurrentText(value[1])
                combo.setProperty("row", row)
                combo.currentTextChanged.connect(self._on_waveform_changed_for_widget)
                self.channel_table.setCellWidget(row, col, combo)
            else:
                text = value[1]
                check = value[2]
                item = QtWidgets.QTableWidgetItem(text)
                if col in (self.COL_INDEP, self.COL_LINK):
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setCheckState(check)
                self.channel_table.setItem(row, col, item)

        if "__state__" in data:
            state_item = self.channel_table.item(row, self.COL_CHANNEL)
            if state_item is not None:
                state_item.setData(QtCore.Qt.UserRole, data["__state__"])

        if row == self.channel_table.currentRow():
            self._load_details_from_row(row)

    def _build_detail_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.detail_title = QtWidgets.QLabel("Channel Details")
        layout.addWidget(self.detail_title)

        # Triangle params
        self.tri_group = QtWidgets.QGroupBox("Triangle Params")
        tri_layout = QtWidgets.QFormLayout(self.tri_group)
        self.tri_start = QtWidgets.QLineEdit("0.0")
        self.tri_first = QtWidgets.QLineEdit("0.0")
        self.tri_second = QtWidgets.QLineEdit("0.0")
        self.tri_dv = QtWidgets.QLineEdit("0.0")
        tri_layout.addRow("Start V", self.tri_start)
        tri_layout.addRow("First Node", self.tri_first)
        tri_layout.addRow("Second Node", self.tri_second)
        tri_layout.addRow("dV", self.tri_dv)

        # Square params
        self.square_group = QtWidgets.QGroupBox("Square Params")
        sq_layout = QtWidgets.QFormLayout(self.square_group)
        self.sq_v_high = QtWidgets.QLineEdit("0.0")
        self.sq_v_low = QtWidgets.QLineEdit("0.0")
        self.sq_n_high = QtWidgets.QLineEdit("10")
        self.sq_n_low = QtWidgets.QLineEdit("10")
        self.sq_n_ramp = QtWidgets.QLineEdit("0")
        self.sq_n_offset = QtWidgets.QLineEdit("0")
        sq_layout.addRow("V High", self.sq_v_high)
        sq_layout.addRow("V Low", self.sq_v_low)
        sq_layout.addRow("n_high", self.sq_n_high)
        sq_layout.addRow("n_low", self.sq_n_low)
        sq_layout.addRow("n_ramp", self.sq_n_ramp)
        sq_layout.addRow("n_offset", self.sq_n_offset)

        # Three-stage square params
        self.square3_group = QtWidgets.QGroupBox("Square-3 Params")
        sq3_layout = QtWidgets.QFormLayout(self.square3_group)
        self.sq3_v_high = QtWidgets.QLineEdit("0.0")
        self.sq3_v_low = QtWidgets.QLineEdit("0.0")
        self.sq3_v_mid = QtWidgets.QLineEdit("0.0")
        self.sq3_n_high = QtWidgets.QLineEdit("10")
        self.sq3_n_low = QtWidgets.QLineEdit("10")
        self.sq3_n_mid = QtWidgets.QLineEdit("10")
        self.sq3_n_offset = QtWidgets.QLineEdit("0")
        sq3_layout.addRow("V High", self.sq3_v_high)
        sq3_layout.addRow("V Low", self.sq3_v_low)
        sq3_layout.addRow("V Mid", self.sq3_v_mid)
        sq3_layout.addRow("n_high", self.sq3_n_high)
        sq3_layout.addRow("n_low", self.sq3_n_low)
        sq3_layout.addRow("n_mid", self.sq3_n_mid)
        sq3_layout.addRow("n_offset", self.sq3_n_offset)

        # Sine params
        self.sine_group = QtWidgets.QGroupBox("Sine Params")
        sine_layout = QtWidgets.QFormLayout(self.sine_group)
        self.sine_v_amp = QtWidgets.QLineEdit("0.0")
        self.sine_v_offset = QtWidgets.QLineEdit("0.0")
        self.sine_n_period = QtWidgets.QLineEdit("100")
        sine_layout.addRow("V Amp", self.sine_v_amp)
        sine_layout.addRow("V Offset", self.sine_v_offset)
        sine_layout.addRow("n_period", self.sine_n_period)

        # Fixed params
        self.fixed_group = QtWidgets.QGroupBox("Fixed Params")
        fixed_layout = QtWidgets.QFormLayout(self.fixed_group)
        self.fixed_v = QtWidgets.QLineEdit("0.0")
        fixed_layout.addRow("V Fixed", self.fixed_v)

        layout.addWidget(self.tri_group)
        layout.addWidget(self.square_group)
        layout.addWidget(self.square3_group)
        layout.addWidget(self.sine_group)
        layout.addWidget(self.fixed_group)

        self.save_detail_btn = QtWidgets.QPushButton("Apply To Selected Channel")
        self.save_detail_btn.clicked.connect(self._on_apply_details)
        layout.addWidget(self.save_detail_btn)
        layout.addStretch(1)

        self._update_detail_visibility("Triangle")
        return panel

    def _on_row_selected(self, current: QtCore.QModelIndex, _prev: QtCore.QModelIndex) -> None:
        if not current.isValid():
            return
        self._load_details_from_row(current.row())

    def _load_details_from_row(self, row: int) -> None:
        state = self._get_row_state(row)
        self.detail_title.setText(f"Channel Details: {state['channel_name']}")
        waveform = state["waveform"]
        self._update_detail_visibility(waveform)

        self.tri_start.setText(str(state["start_voltage"]))
        self.tri_first.setText(str(state["first_node"]))
        self.tri_second.setText(str(state["second_node"]))
        self.tri_dv.setText(str(state["dV"]))

        self.sq_v_high.setText(str(state["v_high"]))
        self.sq_v_low.setText(str(state["v_low"]))
        self.sq_n_high.setText(str(state["n_high"]))
        self.sq_n_low.setText(str(state["n_low"]))
        self.sq_n_ramp.setText(str(state["n_ramp"]))
        self.sq_n_offset.setText(str(state["n_offset"]))

        self.sq3_v_high.setText(str(state["v_high"]))
        self.sq3_v_low.setText(str(state["v_low"]))
        self.sq3_v_mid.setText(str(state["v_mid"]))
        self.sq3_n_high.setText(str(state["n_high"]))
        self.sq3_n_low.setText(str(state["n_low"]))
        self.sq3_n_mid.setText(str(state["n_mid"]))
        self.sq3_n_offset.setText(str(state["n_offset"]))

        self.sine_v_amp.setText(str(state["v_amp"]))
        self.sine_v_offset.setText(str(state["v_offset"]))
        self.sine_n_period.setText(str(state["n_period"]))

        self.fixed_v.setText(str(state["v_fixed"]))

    def _on_apply_details(self) -> None:
        row = self.channel_table.currentRow()
        if row < 0:
            return
        state = self._get_row_state(row)
        waveform = self._get_waveform_value(row)
        state["waveform"] = waveform

        state["start_voltage"] = self.tri_start.text()
        state["first_node"] = self.tri_first.text()
        state["second_node"] = self.tri_second.text()
        state["dV"] = self.tri_dv.text()

        if waveform.lower() == "square-3":
            state["v_high"] = self.sq3_v_high.text()
            state["v_low"] = self.sq3_v_low.text()
            state["n_high"] = self.sq3_n_high.text()
            state["n_low"] = self.sq3_n_low.text()
            state["n_offset"] = self.sq3_n_offset.text()
        else:
            state["v_high"] = self.sq_v_high.text()
            state["v_low"] = self.sq_v_low.text()
            state["n_high"] = self.sq_n_high.text()
            state["n_low"] = self.sq_n_low.text()
            state["n_offset"] = self.sq_n_offset.text()

        state["n_ramp"] = self.sq_n_ramp.text()

        state["v_mid"] = self.sq3_v_mid.text()
        state["n_mid"] = self.sq3_n_mid.text()

        state["v_amp"] = self.sine_v_amp.text()
        state["v_offset"] = self.sine_v_offset.text()
        state["n_period"] = self.sine_n_period.text()

        state["v_fixed"] = self.fixed_v.text()

        self._set_row_state(row, state)

    def _update_detail_visibility(self, waveform: str) -> None:
        wf = waveform.lower()
        self.tri_group.setVisible(wf == "triangle")
        self.square_group.setVisible(wf == "square")
        self.square3_group.setVisible(wf == "square-3")
        self.sine_group.setVisible(wf == "sine")
        self.fixed_group.setVisible(wf == "fixed")

    def _get_row_state(self, row: int) -> dict[str, Any]:
        # Persist per-row detail state on the row item itself.
        item = self.channel_table.item(row, self.COL_CHANNEL)
        if item is None:
            return self._default_row_state(f"row{row}")
        state = item.data(QtCore.Qt.UserRole)
        if not isinstance(state, dict):
            state = self._default_row_state(item.text())
            item.setData(QtCore.Qt.UserRole, state)
        state["channel_name"] = item.text()
        state["waveform"] = self._get_waveform_value(row)
        return state

    def _set_row_state(self, row: int, state: dict[str, Any]) -> None:
        item = self.channel_table.item(row, self.COL_CHANNEL)
        if item is None:
            item = QtWidgets.QTableWidgetItem(state.get("channel_name", f"row{row}"))
            self.channel_table.setItem(row, self.COL_CHANNEL, item)
        item.setData(QtCore.Qt.UserRole, state)

    @staticmethod
    def _default_row_state(channel_name: str) -> dict[str, Any]:
        return {
            "channel_name": channel_name,
            "waveform": "Triangle",
            "start_voltage": "0.0",
            "first_node": "0.0",
            "second_node": "0.0",
            "dV": "0.0",
            "v_high": "0.0",
            "v_low": "0.0",
            "v_mid": "0.0",
            "v_fixed": "0.0",
            "n_high": "10",
            "n_low": "10",
            "n_mid": "10",
            "n_ramp": "0",
            "n_offset": "0",
            "v_amp": "0.0",
            "v_offset": "0.0",
            "n_period": "100",
        }

    def _set_row_state_defaults(self, row: int) -> None:
        item = self.channel_table.item(row, self.COL_CHANNEL)
        if item is None:
            return
        item.setData(QtCore.Qt.UserRole, self._default_row_state(item.text()))

    def _set_indicator(self, state: str) -> None:
        if state == "running":
            color = "#2ecc71"
        elif state == "paused":
            color = "#f1c40f"
        else:
            color = "#95a5a6"
        self.run_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 6px;"
        )

    def _set_run_state(self, running: bool, paused: bool = False) -> None:
        self.run_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        if not running:
            self.run_status.setText("Idle")
            self._set_indicator("idle")
            self.pause_btn.setText("Pause")
        elif paused:
            self.run_status.setText("Paused")
            self._set_indicator("paused")
            self.pause_btn.setText("Resume")
        else:
            self.run_status.setText("Running")
            self._set_indicator("running")
            self.pause_btn.setText("Pause")

    def _on_pause_resume(self) -> None:
        if self.run_worker is None:
            return
        if self.run_worker.is_paused:
            try:
                configs = self._collect_channel_configs()
                dt_list = self._parse_float_list(self.dt_list.text())
                repeat = int(self.repeat.text().strip() or "1")
                round_delay = float(self.round_delay.text().strip() or "0")
                delay_ratio = float(self.delayNPLC_ratio.text().strip() or "0.8")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", str(exc))
                return
            self.run_worker.request_resume(configs, dt_list, repeat, round_delay, delay_ratio)
            self._set_run_state(True, paused=False)
        else:
            self.run_worker.request_pause()
            self._set_run_state(True, paused=True)

    def _on_stop(self) -> None:
        if self.run_worker is None:
            return
        self.run_worker.request_stop()
        self._set_run_state(False)

    def _on_ramp_to_zero(self) -> None:
        if self.station is None or not self.keithleys:
            QtWidgets.QMessageBox.warning(self, "Not Connected", "Connect to keithleys first.")
            return
        try:
            configs = self._collect_channel_configs()
            sweepers, _ = self._build_sweepers(configs)
            for sweeper in sweepers:
                if "nano" not in sweeper["name"]:
                    utilities.ramp_voltage(sweeper["channel"], 0)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Ramp Failed", str(exc))

    def _on_run(self) -> None:
        if self.station is None or not self.keithleys:
            QtWidgets.QMessageBox.warning(self, "Not Connected", "Connect to keithleys first.")
            return

        try:
            configs = self._collect_channel_configs()
            dt_list = self._parse_float_list(self.dt_list.text())
            delay_ratio = float(self.delayNPLC_ratio.text().strip() or "0.8")
            repeat = int(self.repeat.text().strip() or "1")
            round_delay = float(self.round_delay.text().strip() or "0")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", str(exc))
            return

        db_path = self.db_path.text().strip()
        if not db_path:
            QtWidgets.QMessageBox.warning(self, "Missing DB Path", "DB path is required.")
            return

        exp_name = self.exp_name.text().strip() or "gui_experiment"
        device_name = self.device_name.text().strip() or "device"
        csv_path = self.csv_path.text().strip()

        self.run_thread = QtCore.QThread()
        self.run_worker = RunWorker(
            station=self.station,
            keithleys=self.keithleys,
            configs=configs,
            dt_list=dt_list,
            delay_ratio=delay_ratio,
            repeat=repeat,
            round_delay=round_delay,
            db_path=db_path,
            exp_name=exp_name,
            device_name=device_name,
            csv_path=csv_path,
            ramp_up=self.ramp_up.isChecked(),
            ramp_down=self.ramp_down.isChecked(),
            time_independent=self.time_independent.isChecked(),
        )
        self.run_worker.moveToThread(self.run_thread)
        self.run_thread.started.connect(self.run_worker.run)
        self.run_worker.finished.connect(self.run_thread.quit)
        self.run_worker.finished.connect(self.run_worker.deleteLater)
        self.run_thread.finished.connect(self.run_thread.deleteLater)
        self.run_worker.status.connect(self._on_worker_status)
        self.run_worker.error.connect(self._on_worker_error)
        self.run_worker.finished.connect(self._on_worker_finished)

        self._set_run_state(True, paused=False)
        self.run_thread.start()

    def _on_worker_status(self, msg: str) -> None:
        self.run_status.setText(msg)

    def _on_worker_error(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Run Failed", msg)

    def _on_worker_finished(self) -> None:
        self._set_run_state(False)

    @staticmethod
    def _load_yaml_instruments(path: str) -> list[str]:
        if yaml is None:
            raise RuntimeError("PyYAML not installed; install pyyaml to read configs")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        instruments = data.get("instruments", {})
        return list(instruments.keys())


class RunWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        station: Station,
        keithleys: dict[str, Any],
        configs: list[ChannelConfig],
        dt_list: list[float],
        delay_ratio: float,
        repeat: int,
        round_delay: float,
        db_path: str,
        exp_name: str,
        device_name: str,
        csv_path: str,
        ramp_up: bool,
        ramp_down: bool,
        time_independent: bool,
    ) -> None:
        super().__init__()
        self.station = station
        self.keithleys = keithleys
        self.configs = configs
        self.dt_list = dt_list
        self.delay_ratio = delay_ratio
        self.repeat = repeat
        self.round_delay = round_delay
        self.db_path = db_path
        self.exp_name = exp_name
        self.device_name = device_name
        self.csv_path = csv_path
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.time_independent = time_independent

        self._pause_event = threading.Event()
        self._pause_event.set()
        self._rebuild_on_resume = False
        self._step_index = 0
        self.is_paused = False
        self._stop_requested = False
        self._last_volt: tuple[float, ...] | None = None

    @QtCore.pyqtSlot()
    def request_pause(self) -> None:
        self.is_paused = True
        self._pause_event.clear()
        self.status.emit("Paused")

    def request_resume(
        self,
        configs: list[ChannelConfig],
        dt_list: list[float],
        repeat: int,
        round_delay: float,
        delay_ratio: float,
    ) -> None:
        self.configs = configs
        self.dt_list = dt_list
        self.repeat = repeat
        self.round_delay = round_delay
        self.delay_ratio = delay_ratio
        self._rebuild_on_resume = True
        self.is_paused = False
        self._pause_event.set()
        self.status.emit("Running")

    def request_stop(self) -> None:
        self._stop_requested = True
        self._pause_event.set()

    def run(self) -> None:
        try:
            self.status.emit("Running")
            initialise_or_create_database_at(self.db_path)
            test_exp = load_or_create_experiment(
                experiment_name=self.exp_name,
                sample_name=self.device_name,
            )

            sweepers, sweepers_save_order = self._build_sweepers(self.configs)
            meas_forward, time_param, _indep = utilities.setup_database_registers_arb(
                self.station,
                test_exp,
                sweepers_save_order,
                time_independent=self.time_independent,
            )
            meas_forward.write_period = 2

            if self.ramp_up:
                for sweeper in sweepers:
                    utilities.ramp_voltage(sweeper["channel"], sweeper["v_range"][0])

            time_param.reset_clock()

            for sweeper in sweepers:
                ch = sweeper["channel"]
                trigger_fns.source_trig_params(ch)
                trigger_fns.meas_trig_params(ch)

            channels = [s["channel"] for s in sweepers]

            plan = self._build_plan(self.configs, self.dt_list, self.repeat, self.round_delay)
            last_dt = None

            with meas_forward.run() as forward_saver:
                while self._step_index < len(plan):
                    if self._stop_requested:
                        break
                    self._pause_event.wait()

                    if self._rebuild_on_resume:
                        sweepers, sweepers_save_order = self._build_sweepers(self.configs)
                        channels = [s["channel"] for s in sweepers]
                        plan = self._build_plan(
                            self.configs, self.dt_list, self.repeat, self.round_delay
                        )
                        if self._last_volt is not None:
                            resume_idx = self._find_resume_index(plan, self._last_volt)
                            if resume_idx is not None:
                                self._step_index = resume_idx
                        if self._step_index >= len(plan):
                            break
                        self._rebuild_on_resume = False

                    entry = plan[self._step_index]
                    if entry["type"] == "sleep":
                        if self._stop_requested:
                            break
                        threading.Event().wait(entry["seconds"])
                        self._step_index += 1
                        continue

                    dt_in = entry["dt"]
                    if last_dt is None or dt_in != last_dt:
                        self._set_ktime(sweepers_save_order, dt_in, self.delay_ratio)
                        last_dt = dt_in

                    for x, sweeper in zip(entry["volt"], sweepers):
                        trigger_fns.set_v(sweeper["channel"], x)

                    t = time_param()
                    get_readings = []
                    independent_params = []

                    trigger_fns.trigger(list(self.keithleys.values()), channels)

                    for sweeper in sweepers_save_order:
                        v, j = trigger_fns.recall_buffer(sweeper["channel"])
                        v = float(v)
                        j = float(j)
                        get_readings.append((sweeper["channel"].curr, j))

                        if sweeper["independent"]:
                            independent_params.append((sweeper["channel"].volt, v))
                        else:
                            get_readings.append((sweeper["channel"].volt, v))

                        if "temperature" in sweeper["name"]:
                            temperature = utilities.rToT(v / j) if j != 0 else 0.0
                            get_readings.append((sweeper["channel"].temperature, temperature))

                    forward_saver.add_result(
                        *independent_params,
                        *get_readings,
                        (time_param, t),
                    )
                    self._last_volt = entry["volt"]
                    self._step_index += 1
                    if self._stop_requested:
                        break

            data_forward = forward_saver.dataset
            if self.csv_path:
                csv_file = ArbitrarySweeperGUI._resolve_csv_path(
                    self.csv_path, self.device_name, self.exp_name, data_forward.run_id
                )
                data_forward.to_pandas_dataframe().to_csv(csv_file)

            if self.ramp_down:
                for sweeper in sweepers:
                    if "nano" not in sweeper["name"]:
                        utilities.ramp_voltage(sweeper["channel"], 0)

            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit()

    @staticmethod
    def _set_ktime(sweepers: list[dict[str, Any]], dt_in: float, delay_ratio: float) -> None:
        nplc_set = dt_in * 50 * (1 - delay_ratio)
        delay = dt_in - (nplc_set / 50)
        for sweeper in sweepers:
            sweeper["channel"].delay(delay)
            sweeper["channel"].nplc(nplc_set)

    @staticmethod
    def _build_v_range(cfg: ChannelConfig) -> np.ndarray:
        if cfg.waveform.lower() == "square":
            return RunWorker._build_square_wave(cfg)
        if cfg.waveform.lower() == "square-3":
            return RunWorker._build_square3_wave(cfg)
        if cfg.waveform.lower() == "sine":
            return RunWorker._build_sine_wave(cfg)
        if cfg.waveform.lower() == "fixed":
            return np.array([cfg.v_fixed], dtype=float)

        if cfg.dV == 0:
            return np.array([cfg.start_voltage], dtype=float)

        n = 1 + abs(int((cfg.first_node - cfg.start_voltage) / cfg.dV))
        v_range1 = np.linspace(cfg.start_voltage, cfg.first_node, n)[:-1]
        n = 1 + abs(int((cfg.second_node - cfg.first_node) / cfg.dV))
        v_range2 = np.linspace(cfg.first_node, cfg.second_node, n)[:-1]
        n = 1 + abs(int((cfg.start_voltage - cfg.second_node) / cfg.dV))
        v_range3 = np.linspace(cfg.second_node, cfg.start_voltage, n)
        return np.concatenate((v_range1, v_range2, v_range3))

    @staticmethod
    def _build_square_wave(cfg: ChannelConfig) -> np.ndarray:
        n_high = max(0, int(cfg.n_high))
        n_low = max(0, int(cfg.n_low))
        n_ramp = max(0, int(cfg.n_ramp))
        v_high = float(cfg.v_high)
        v_low = float(cfg.v_low)
        n_offset = int(cfg.n_offset)

        if n_ramp > 0:
            ramp_up = np.linspace(v_low, v_high, n_ramp + 2)[1:-1]
            ramp_down = np.linspace(v_high, v_low, n_ramp + 2)[1:-1]
        else:
            ramp_up = np.array([], dtype=float)
            ramp_down = np.array([], dtype=float)

        high = np.full(n_high, v_high, dtype=float)
        low = np.full(n_low, v_low, dtype=float)

        cycle = np.concatenate((low, ramp_up, high, ramp_down, low))
        if cycle.size == 0:
            return np.array([v_low], dtype=float)

        shift = n_offset % cycle.size
        if shift:
            cycle = np.concatenate((cycle[shift:], cycle[:shift]))
        return cycle

    @staticmethod
    def _build_square3_wave(cfg: ChannelConfig) -> np.ndarray:
        v_high = float(cfg.v_high)
        v_low = float(cfg.v_low)
        v_mid = float(cfg.v_mid)
        n_high = max(0, int(cfg.n_high))
        n_low = max(0, int(cfg.n_low))
        n_mid = max(0, int(cfg.n_mid))
        n_offset = int(cfg.n_offset)

        mid = np.full(n_mid, v_mid, dtype=float)
        low = np.full(n_low, v_low, dtype=float)
        high = np.full(n_high, v_high, dtype=float)

        cycle = np.concatenate((mid, low, mid, high))
        if cycle.size == 0:
            return np.array([v_mid], dtype=float)

        shift = n_offset % cycle.size
        if shift:
            cycle = np.concatenate((cycle[shift:], cycle[:shift]))
        return cycle

    @staticmethod
    def _build_sine_wave(cfg: ChannelConfig) -> np.ndarray:
        v_amp = float(cfg.v_amp)
        v_offset = float(cfg.v_offset)
        n_period = max(1, int(cfg.n_period))
        t = np.linspace(0, 2 * np.pi, n_period, endpoint=False)
        return v_offset + v_amp * np.sin(t)

    def _build_sweepers(
        self, configs: list[ChannelConfig]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sweepers: list[dict[str, Any]] = []
        for cfg in configs:
            inst_name, ch_name = cfg.channel_name.split(".")
            channel = getattr(self.keithleys[inst_name], ch_name)
            v_range = self._build_v_range(cfg)
            sweepers.append(
                {
                    "channel": channel,
                    "name": cfg.name,
                    "first_node": cfg.first_node,
                    "second_node": cfg.second_node,
                    "start_voltage": cfg.start_voltage,
                    "dV": cfg.dV,
                    "independent": cfg.independent,
                    "v_range": v_range,
                }
            )
        return sweepers, list(sweepers)

    @staticmethod
    def _build_groups(configs: list[ChannelConfig]) -> list[list[int]]:
        groups: list[list[int]] = []
        current: list[int] = []
        for idx, cfg in enumerate(configs):
            current.append(idx)
            if not cfg.link_next:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _build_plan(
        configs: list[ChannelConfig],
        dt_list: list[float],
        repeat: int,
        round_delay: float,
    ) -> list[dict[str, Any]]:
        v_ranges = [RunWorker._build_v_range(cfg) for cfg in configs]
        groups = RunWorker._build_groups(configs)

        group_iters: list[list[tuple[float, ...]]] = []
        for group in groups:
            group_ranges = [v_ranges[i] for i in group]
            max_len = max(len(r) for r in group_ranges)
            padded = [
                np.pad(r, (0, max_len - len(r)), mode="edge") for r in group_ranges
            ]
            group_iters.append(list(zip(*padded)))

        plan: list[dict[str, Any]] = []
        for dt_in in dt_list:
            for _rep in range(repeat):
                for combo in itertools.product(*group_iters):
                    flat = [0.0] * len(v_ranges)
                    for group, values in zip(groups, combo):
                        for idx, val in zip(group, values):
                            flat[idx] = val
                    plan.append({"type": "measure", "dt": dt_in, "volt": tuple(flat)})
                if round_delay > 0:
                    plan.append({"type": "sleep", "seconds": round_delay})
        return plan

    @staticmethod
    def _find_resume_index(
        plan: list[dict[str, Any]], last_volt: tuple[float, ...]
    ) -> int | None:
        best_idx = None
        best_dist = None
        for idx, entry in enumerate(plan):
            if entry.get("type") != "measure":
                continue
            volt = entry.get("volt")
            if volt is None or len(volt) != len(last_volt):
                continue
            dist = sum((a - b) ** 2 for a, b in zip(volt, last_volt))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = ArbitrarySweeperGUI()
    win.resize(1100, 800)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
