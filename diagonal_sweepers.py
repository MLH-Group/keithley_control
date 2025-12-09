import numpy as np
import qcodes as qc
from qcodes.dataset import (
    Measurement,
    experiments,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
)

from typing import Optional

## Dummy instruments for generating synthetic data
from qcodes.tests.instrument_mocks import (
    DummyInstrument,
    DummyInstrumentWithMeasurement
)

from time import sleep

## Multidimensional scanning module
from qcodes.utils.dataset.doNd import (
    dond,
    do1d,
    do2d,
    LinSweep,
    ArraySweep
)

from qcodes.instrument.parameter import (
    Parameter
)

from qcodes.instrument.base import (
    InstrumentBase
)

## Integrated plotting module
from qcodes.dataset.plotting import plot_dataset

## Using interactive widget
from qcodes.interactive_widget import experiments_widget

from qcodes.station import Station

import os


from qcodes.instrument.specialized_parameters import ElapsedTimeParameter

class DiagonalSource(Parameter):
    def __init__(self, name, source1, source2, slope=-1, intercept=0):
        self._source1 = source1
        self._source2 = source2
        self._slope = slope
        self._intercept = intercept
        super().__init__(name)

    @property
    def underlying_instrument(self) -> Optional[InstrumentBase]:
        return self._dac_param.root_instrument

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = value

    @property
    def intercept(self):
        return self._intercept

    @intercept.setter
    def intercept(self, value):
        self._intercept = value

    def get_raw(self):
        return self._source2.get()

    def set_raw(self, x):
        y = self._intercept + self._slope*x
        self._source1.set(x)
        self._source2.set(y)

class DiagonalIntercept(Parameter):
    def __init__(self, name, diagonal_source, intercept=0):
        self._intercept = intercept
        self._source = diagonal_source
        super().__init__(name)

    @property
    def underlying_instrument(self) -> Optional[InstrumentBase]:
        return self._dac_param.root_instrument

    def get_raw(self):
        return self._intercept

    def set_raw(self, y):
        self._intercept = y
        self._source.intercept = y
