from qcodes.instrument import (
    VisaInstrument,
    Parameter
)
from qcodes.utils import validators as vals
import numpy as np

class keithley2182a(VisaInstrument):
    """
    Class to represent a Keithley2182a nanovoltmeter
    Not using InstrumentChannel classes since ch1 and ch2 do not have equal behaviour
    """
    def __init__(self, name: str, address: str, **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA ressource address
        """
        super().__init__(name, address, terminator="\n", **kwargs)

        self.volt = Parameter(
            "volt",
            unit="V",
            label = f'Voltage<{name}>',
            get_cmd=":SENS:DATA:FRES?",
            get_parser=float,
            snapshot_get=False,
            instrument=self
        )

        # self.volt = Parameter(
        #     "volt",
        #     unit="V",
        #     get_cmd=":MEASure:VOLTage[:DC]?",
        #     get_parser=float,
        #     snapshot_get=False,
        #     instrument=self
        # )
        

        # self.channel = Parameter(
        #     "channel",
        #     get_cmd=":SENS:CHAN?",
        #     set_cmd=":SENS:CHAN {:d}",
        #     vals=vals.Enum(*[0, 1,2]),
        #     get_parser=int,
        #     instrument=self
        # )

        self.nplc = Parameter(
            "nplc",
            unit = "plc",
            get_cmd=":SENS:VOLT:NPLC?",
            set_cmd = ":SENS:VOLT:NPLC {:f}",
            get_parser=float,
            instrument=self
        )

        # self.trigger_source = Parameter(
        #     "trigger_source",
        #     get_cmd=":TRIG:SOUR?",
        #     set_cmd=":TRIG:SOUR {:s}",
        #     val_mapping={"immediate": "IMM", "timer": "TIM", "manual": "MAN", "bus": "BUS", "external": "EXT"},
        #     instrument=self
        # )

        # self.buffer_points = Parameter(
        #     "buffer_points",
        #     get_cmd=":TRAC:POIN?",
        #     set_cmd = ":TRAC:POIN {:d}",
        #     get_parser=int,
        #     instrument=self
        # )

        # self.sample_count = Parameter(
        #     "sample_count",
        #     get_cmd=":SAMP:COUN?",
        #     set_cmd = ":SAMP:COUN {:d}",
        #     get_parser=int,
        #     instrument=self
        # )

        # self.data = Parameter(
        #     "buffer_data",
        #     get_cmd=":TRAC:DATA?",
        #     get_parser=lambda s: np.fromstring(s, sep=",", dtype=np.float64),
        #     instrument=self
        # )

        # self.trigger_delay = Parameter(
        #     "trigger_delay",
        #     get_cmd=":TRIG:DEL?",
        #     set_cmd = ":TRIG:DEL {:f}",
        #     get_parser=float,
        #     instrument=self
        # )

        # self.store_readings = Parameter(
        #     "store_readings",
        #     get_cmd=":TRAC:FEED:CONT?",
        #     set_cmd=":TRAC:FEED:CONT {:s}",
        #     val_mapping={False: "NEV", True: "NEXT"},
        #     instrument=self
        # )

        self.connect_message()

    def reset(self):
        self.write("*RST")

    def clear_buffer(self):
        self.write("TRAC:CLE")

    def trigger(self):
        self.write("*TRG")

    def arm(self):
        self.write(":INIT")

