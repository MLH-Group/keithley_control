#For sweeping E field for fewer number (sparse_N) of doping values. Can also truncate, to avoid large doping values 
#but still have large E field range. Cuts corners off rotated square
from import_all import *

def sparse_intercepts(intercepts, sweep_points, sparse_N, truncate_N = 0):
    dN = int(len(intercepts)/(sparse_N-1))
    if truncate_N == 0:
        sparse_intercepts = intercepts[0::dN]
        sparse_sweep_points = sweep_points[0::dN]
    else:
        sparse_intercepts = intercepts[0::dN][truncate_N:-truncate_N]
        sparse_sweep_points = sweep_points[0::dN][truncate_N:-truncate_N]

    return sparse_intercepts, sparse_sweep_points


def cutE(diag, intercept, intercepts, sweep_points, threshold):
    sweep_points_new = []
    for inter, xrange in zip(intercepts, sweep_points):
        intercept(inter)
        xrange_new = []
        for x in xrange:
            y = diag.slope*x + intercept()
            if abs(round(x-y,2)) > threshold:
                continue
            else:
                xrange_new.append(x)
        sweep_points_new.append(xrange_new)
    return sweep_points_new

#Ramp voltage to designated value, slowly
def ramp_voltage(channel, final, rampdV = 5e-5, rampdT = 1e-3): 
    initial = channel.volt()
    ramp = np.linspace(initial, final, int(1+abs((initial-final)/rampdV)))
    print(f'ramping {channel} from {initial} to {final}')
    for x in ramp:
        channel.volt(x)
        sleep(rampdT)

def pad_arrays(a, b):
    len_a, len_b = len(a), len(b)
    
    if len_a < len_b:
        # Pad array 'a' with its last value
        a = np.pad(a, (0, len_b - len_a), 'edge')
    elif len_b < len_a:
        # Pad array 'b' with its last value
        b = np.pad(b, (0, len_a - len_b), 'edge')
    
    return a, b

#Ramp two channels simultaneously to designated values. First resets to 0.
def ramp_two_voltage(channel1, channel2, final1, final2, reset = True, x_initial = 0, y_initial = 0, rampdV = 5e-5, initial = 0, rampdT = 1e-3):
    # first make sure both channels are at zero
    if reset:
        ramp_voltage(channel1, x_initial, rampdV = rampdV, rampdT =rampdT)
        ramp_voltage(channel2, y_initial, rampdV = rampdV, rampdT =rampdT)
    
    ramp1 = np.linspace(x_initial, final1, int(1+abs((x_initial-final1)/rampdV)))
    ramp2 = np.linspace(y_initial, final2, int(1+abs((y_initial-final2)/rampdV)))

    ramp1, ramp2 = pad_arrays(ramp1, ramp2)

    print(f'ramping {channel1} and {channel2} to {final1} and {final2}')
    for x1, x2 in zip(ramp1, ramp2):
        channel1.volt(x1)
        channel2.volt(x2)
        sleep(rampdT)


def setup_database_registers_nSweep(station, source_meter_sweeping, source_meter_fixed, test_exp):
    #Set up database registers
    time = ElapsedTimeParameter("time")
    meas_forward = Measurement(exp=test_exp, station=station, name='forward')
    # Register sweep voltages as dependent parameters
    meas_forward.register_parameter(source_meter_sweeping.smua.volt)
    # meas_forward.register_parameter(source_meter_sweeping.smub.volt)
    # Register currents as dependent parameters
    meas_forward.register_parameter(source_meter_sweeping.smua.curr, setpoints=(source_meter_sweeping.smua.volt,))
    meas_forward.register_parameter(source_meter_sweeping.smub.curr, setpoints=(source_meter_sweeping.smua.volt,))
    meas_forward.register_parameter(source_meter_sweeping.smub.volt, setpoints=(source_meter_sweeping.smua.volt,))
    meas_forward.register_parameter(source_meter_fixed.smub.volt, setpoints=(source_meter_sweeping.smua.volt,))
    meas_forward.register_parameter(source_meter_fixed.smub.curr, setpoints=(source_meter_sweeping.smua.volt,))
    meas_forward.register_parameter(time, setpoints=(source_meter_sweeping.smua.volt,))

    return meas_forward, time


def setup_database_registers_arb(station, test_exp, sweepers, time_independent = False):
    #Set up database registers
    time = ElapsedTimeParameter("time")
    meas_forward = Measurement(exp=test_exp, station=station, name='forward')
    #meas_back = Measurement(exp=test_exp, station=station, name='back')
    # Register sweep voltages as dependent parameters

    independent_params = []
    for sweeper in sweepers:
        channel = sweeper["channel"]
        if sweeper["independent"]:
            meas_forward.register_parameter(channel.volt)
            independent_params.append(channel.volt)
    # Register currents as dependent parameters
            
    if time_independent:
        meas_forward.register_parameter(time)
        independent_params.append(time)

    for sweeper in sweepers:
        channel = sweeper["channel"]
        if not "nano" in sweeper["name"] or "temperature" in sweeper["name"]:
            meas_forward.register_parameter(channel.curr, setpoints = (*independent_params,))
        if "temperature" in sweeper["name"]:
            meas_forward.register_parameter(channel.temperature, setpoints = (*independent_params,))
       # meas_back.register_parameter(channel.curr, (*independent_params,))
        if not sweeper["independent"]:
            meas_forward.register_parameter(channel.volt, setpoints = (*independent_params,))
            #meas_back.register_parameter(channel.volt, (*independent_params,))

    if not time_independent:
        meas_forward.register_parameter(time, setpoints=(*independent_params,))
    
    return meas_forward, time, independent_params



def setup_database_registers_inPlane(station, test_exp, sweepers):
    #Set up database registers
    time = ElapsedTimeParameter("time")
    meas_forward = Measurement(exp=test_exp, station=station, name='forward')
    meas_back = Measurement(exp=test_exp, station=station, name='back')

    #meas_back = Measurement(exp=test_exp, station=station, name='back')
    # Register sweep voltages as dependent parameters

    independent_params = []
    for sweeper in sweepers:
        channel = sweeper["channel"]
        if sweeper["independent"]:
            meas_forward.register_parameter(channel.volt)
            meas_back.register_parameter(channel.volt)
            independent_params.append(channel.volt)
    # Register currents as dependent parameters
            
    
    for sweeper in sweepers:
        channel = sweeper["channel"]
        
        meas_forward.register_parameter(channel.curr, setpoints = (*independent_params,))
        meas_back.register_parameter(channel.curr, setpoints = (*independent_params,))
        if not sweeper["independent"]:
            meas_forward.register_parameter(channel.volt, setpoints = (*independent_params,))
            meas_back.register_parameter(channel.volt, (*independent_params,))
        
    
    meas_forward.register_parameter(time, setpoints=(*independent_params,))
    meas_back.register_parameter(time, setpoints=(*independent_params,))
    
    return meas_forward, meas_back, time


def record_data(x,y,time, sweepa, sweepb, fixed, saver):
    get_i1 = sweepa.curr()
    get_i2 = sweepb.curr()
    get_i3 = fixed.curr()
    get_v3 = fixed.volt()
    t = time()
                
    saver.add_result(
        (sweepa.volt, x),
        (sweepb.volt, y),
        (sweepa.curr, get_i1),
        (sweepb.curr, get_i2),
        (fixed.curr, get_i3),
        (fixed.volt, get_v3),
        (time, t),
        )
    

def record_data4E(x,y,time, sweepa, sweepb, fixed,fixed2, saver):
    get_i1 = sweepa.curr()
    get_i2 = sweepb.curr()
    get_i3 = fixed.curr()
    get_v3 = fixed.volt()
    get_i4 = fixed2.curr()
    get_v4 = fixed2.volt()
    t = time()
                
    saver.add_result(
        (sweepa.volt, x),
        (sweepb.volt, y),
        (sweepa.curr, get_i1),
        (sweepb.curr, get_i2),
        (fixed.curr, get_i3),
        (fixed.volt, get_v3),
        (fixed2.curr, get_i4),
        (fixed2.volt, get_v4),
        (time, t),
        )
    


def record_data_arb(time, sweepers, saver):
    result = []
    for sweeper in sweepers:
        chan = sweeper["channel"]
        result.append((chan.volt, chan.volt()))
    
    for sweeper in sweepers:
        chan = sweeper["channel"]
        if not "nano_volt" in sweeper["name"]:
            result.append((chan.curr, chan.curr()))

    t = time()
    result.append((time, t))
    saver.add_result(*result)



def calcXY(eField, n):
    x = (n+eField)/2
    y = (n-eField)/2

    return x,y


def plot_diagonals(intercepts, sweep_points, diag, intercept):
    vtop = []
    vbottom = []

    for inter, xrange in zip(intercepts, sweep_points):
            intercept(inter)
            for x in xrange:
                y = diag.slope*x + intercept()
                vbottom.append(y)
                vtop.append(x)
            for x in xrange[::-1]:
                y = diag.slope*x + intercept()
                vbottom.append(y)
                vtop.append(x)

    vbottom = np.array(vbottom)
    vtop = np.array(vtop)
    print(f'Total time = {(len(vtop))/3600} hrs')
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    ax1.plot(vtop, marker = '.')
    ax2.plot(vbottom, marker = '.')
    ax3.plot(vtop-vbottom, marker = '.')
    print(max(vtop-vbottom))
    ax4.plot(vtop + vbottom, marker = '.')

    ax1.set_ylabel('V Top (V)')
    ax2.set_ylabel('V Bottom (V)')
    ax3.set_ylabel('E (V)')
    ax4.set_ylabel('n (V)')

    fig.tight_layout()



def rToT(r):
    r = abs(r)
    A1 = .003354
    B1 = 3e-4
    C1 = 5.09E-6
    D1 = 2.19e-07
    R25 = 1e4
    if r > 0:
        denom = A1 + B1*np.log(r/R25) + C1*np.log(r/R25)**2 + D1*np.log(r/R25)**3
    else:
        print("Invalid r value, returning 0")
        return 0
    t = -273.15 + 1/denom
    return t