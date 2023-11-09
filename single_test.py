
import os
import sys

sys.path.append(os.path.join("./", "common"))
import sfr_uzf_mvr_support_funcs as sageBld

import flopy
import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
from figspecs import USGSFigure
import random
sys.path.append(os.path.join("./TLnewtest2sfb"))

mf6exe = os.path.abspath("/bsuhome/hangchen/modcali6/mf6")

data = pd.read_csv('S5.csv')
index = 3164


K_layer1 = data['parK_layer1'][index]
K_layer2 = data['parK_layer2'][index]
K_layer3 = data['parK_layer3'][index]
porosity1 = data['parporosity1'][index]
porosity2 = data['parporosity2'][index]
porosity3 = data['parporosity3'][index]
rate = data['parrate'][index]
ani = data['parani'][index]#*0.9999
ani2 = data['parani2'][index]



config.base_ws = './Field1'

ws = config.base_ws
example_name = "TLnewtest2sfb2"
length_units = "meters"
time_units = "days"
nlay = 3  # Number of layers in parent model

coeff1 = 4
coeff2 = 4.5
coeff3 = 5.0

# Number of columns in parent model
delr = 5  # Parent model column width ($m$)
delc = 5 # Parent model row width ($m$

# Time related variables
num_ts = 365 + 366+ 365 + 365 + (365 + 366)*2
perlen = [1]*num_ts
nper = len(perlen)
nstp = [1] * num_ts
tsmult = [1.0] * num_ts


# from mf-nwt .dis file
dat_pth = example_name
top = np.loadtxt(os.path.join(dat_pth, "top.txt"))
regdep = np.loadtxt(os.path.join(dat_pth, "regdep_new.txt"))
fradep = np.loadtxt(os.path.join(dat_pth, "fradep_new.txt"))
idomain1 = np.loadtxt(os.path.join(dat_pth, "id.txt"))



regdep[regdep<=0.1] = 0.15
fradep[fradep<=regdep+0.1] = regdep[fradep<=regdep+0.1] + 0.15


bot1 = top - regdep
bot2 = top - fradep

dis =  bot2.copy()
dis[idomain1 ==0] = np.nan
bot3 = dis - 30

# from mf-nwt .bas file
for i in range(1000)[300:]:

    strt = bot1 - float(i)/100
    strt2 = bot1 - float(i)/100
    strt3 = bot1 - float(i)/100
    print(float(i)/100)
    nrow = top.shape[0]  # Number of rows in parent model
    ncol = top.shape[1]



    idomain = np.abs(idomain1)

    # from mf-nwt .upw file
    theta_res = 0.025

    k11 = np.ones((top.shape))*10**K_layer1 # L/T
    sy = np.ones((top.shape))*(porosity1-theta_res)
    sy2 = np.ones((top.shape))*(porosity2-theta_res)
    sy3 = np.ones((top.shape))*(porosity3-theta_res)

    k33 = k11


    icelltype = 1  # Water table resides in layer 1
    iconvert = np.ones_like(strt)

    import pandas as pd
    top_sfr = np.zeros((top.shape))
    df = pd.read_csv(os.path.join(dat_pth, "sfr_pack.txt"), header=None, delimiter=r"\s+")

    for i in range(len(df)):
        top_sfr[df[2][i]-1,df[3][i]-1] = df[7][i]

    for i in range(len(df)):
        df[7][i] = top[df[2][i]-1,df[3][i]-1]

    for i in range(len(df)-1):
        temp = np.array(np.abs(df[7][i]-df[7][i+1])/df[4][i])
        if temp >0:
            df[6][i] = temp
        else:
            df[6][i] = 1e-5


    df[6][len(df)-1] = np.array(df[6][len(df)-2])
    for i in range(len(df)):
        df[5][i] = 5*df[5][i]
        df[8][i] = df[8][i]

    rwid = 3.0
    rbth = 1.0
    rhk = 10**K_layer1
    man = 0.035
    ustrf = 1.0
    ndv = 0
    pkdat = []
    sfrcells = []
    sfrcells2 = []
    for i in np.arange(len(df)):

        pkdat.append(
                (
                i,
                (df[1][i]-1,df[2][i]-1,df[3][i]-1),
                df[4][i],
                df[5][i],
                df[6][i],
                df[7][i]-0.05,
                df[8][i],
                rhk,
                man,
                df[11][i],
                ustrf,
                ndv,
                )
            )
        sfrcells.append((df[1][i]-1,df[2][i]-1,df[3][i]-1))
        sfrcells2.append((df[1][i],df[2][i],df[3][i]))

    conns = []
    df2 = np.loadtxt(os.path.join(dat_pth, "sfr_con.txt"))

    for i in range(len(df2)):
        if i == 0:
            temp = [int(df2[i,0]) - 1,int(df2[i,1]) + 1]
        elif i == len(df2)-1:
            temp = [int(df2[i,0]) - 1,int(df2[i,1]) - 1]
        else:
            temp = [int(df2[i,0]) - 1,int(df2[i,1]) - 1,int(df2[i,2]) +  1]
        conns.append(temp)

    # Instantiating MODFLOW 6 drain package
    # Here, the drain (DRN) package is used to simulate groundwater discharge to
    # land surface to keep this water separate from rejected infiltrated simulated
    # by the UZF package. Need to cycle through all land surface cells and create a
    # drain for handling groundwater discharge to land surface
    drn_spd = []
    drn_dict = {}
    drn_dict_rev = {}
    # Use an arbitrarily high conductance term to avoid impeding groundwater disch.
    cond = 10000


    # See definition of auxdepthname in DRN package doc to learn more about param
    ddrn = 0
    idrnno = 0
    for i in np.arange(0, top.shape[0]):
        for j in np.arange(0, top.shape[1]):
            # Don't add drains to sfr and chd cells:
            sfrCell_bool = (
                1
                if len([itm for itm in sfrcells if itm[1] == i and itm[2] == j])
                > 0
                else 0
            )

            if idomain1[i, j] and not sfrCell_bool:
                drn_spd.append([(0, i, j), top[i, j], cond, ddrn])  #  'ddrn',
                # append dictionary of drain indices
                drn_dict.update({(i, j): idrnno})
                drn_dict_rev.update({idrnno: (i, j)})
                idrnno += 1

    ################################
    ##  GHD to discharge the deep groundwater


    ghd_spd = []

    # Use an arbitrarily high conductance term to avoid impeding groundwater disch.
    cond = 10**rate

    # See definition of auxdepthname in DRN package doc to learn more about param

    ighdno = 0
    for i in np.arange(0, top.shape[0]):
        for j in np.arange(0, top.shape[1]):

            if idomain1[i, j] :
                ghd_spd.append([(2, i, j), bot3[i,j],cond])  #  'ddrn',
                # append dictionary of drain indices
                ighdno += 1

    # Prepping input for UZF package
    # Package_data information

    iuzbnd = idomain.copy()

    thts = np.zeros((3,k33.shape[0],k33.shape[1]))

    thts[0] = np.ones((top.shape))*porosity1
    thts[1] = np.ones((top.shape))*porosity2
    thts[2] = np.ones((top.shape))*porosity3


    uzk33 = np.zeros((3,k33.shape[0],k33.shape[1]))
    uzk33[0] =10**K_layer1
    uzk33[1] = 10**K_layer1/10**K_layer2
    uzk33[2] =  10**K_layer1/10**K_layer2/10**K_layer3


    initwc =  np.zeros((3,k33.shape[0],k33.shape[1]))
    initwc[0] = np.ones((k33.shape[0],k33.shape[1]))*(theta_res*4)
    initwc[1] = np.ones((k33.shape[0],k33.shape[1]))*(theta_res*4)
    initwc[2] = np.ones((k33.shape[0],k33.shape[1]))*(theta_res)

    em = np.zeros((3,k33.shape[0],k33.shape[1]))
    em[0] = np.ones((k33.shape[0],k33.shape[1]))*coeff1
    em[1] = np.ones((k33.shape[0],k33.shape[1]))*coeff2
    em[2] = np.ones((k33.shape[0],k33.shape[1]))*coeff3

    #uzk33 = uzk33*2.0
    # uzk33 = np.ones((uzk33.shape))*0.2

    finf_grad = np.ones((top.shape))
    # next, load time series of multipliers
    uz_ts = np.loadtxt(os.path.join(dat_pth, "TLoutflow_11_15.txt"))
    uz_ts1 = np.loadtxt(os.path.join(dat_pth, "TLoutflow_11_15_CC4.txt"))
    uz_ts =  np.hstack((uz_ts[:365 + 366 + 365 +365],uz_ts[:365 + 366],uz_ts[:365 + 366],uz_ts[:365 + 366],uz_ts[:365 + 366]))
    
#    uz_ts = np.hstack((uz_ts[:365 + 366 + 365 +365],uz_ts[:365 + 366],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365],uz_ts1[365 + 366:365 + 366 + 365]
#                       ))


    uz_et = np.loadtxt(os.path.join(dat_pth, "TLET_11_15.txt"))
    uz_et1 = np.loadtxt(os.path.join(dat_pth, "TLET_11_15_CC4.txt"))
    uz_et = np.hstack((uz_et[:365 + 366 + 365 + 365], uz_et[:365 + 366],uz_et[:365 + 366],uz_et[:365 + 366],uz_et[:365 + 366]))
    
#    uz_et = np.hstack((uz_et[:365 + 366 + 365 + 365],uz_et[:365 + 366],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365],uz_et1[365 + 366:365 + 366 + 365]
#                       ))

    uzf_packagedata = []
    pd0 = []
    iuzno_cell_dict = {}
    iuzno_dict_rev = {}
    iuzno = 0
    surfdep = 0.05

    thtr = theta_res

    # Set up the UZF static variables
    nuzfcells = 0
    for k in range(nlay):
        for i in range(0, iuzbnd.shape[0] ):
            for j in range(0, iuzbnd.shape[1] ):
                if iuzbnd[i, j] != 0:
                    nuzfcells += 1
                    if k == 0: # tell if it is the first layer
                        lflag = 1
                        # establish new dictionary entry for current cell
                        # addresses & iuzno connections are both 0-based
                        iuzno_cell_dict.update({(i, j): iuzno})
                        # For post-processing the mvr output, need reverse dict
                        iuzno_dict_rev.update({iuzno: (i, j)})
                    else:
                        lflag = 0

                    # Set the vertical connection, which is the cell below,
                    # but in this 1 layer model set to -1 which flopy adjusts to 0
                    if k == nlay-1:
                         ivertcon = -1 # we need to set as 1, if we have more than 1 layer
                    else:
                         ivertcon = nuzfcells + np.sum(iuzbnd) - 1 # we need to set as 1, if we have more than 1 layer

                    vks = uzk33[k, i, j] # change Kz
                    thtr = thtr
                    thtsx = thts[k, i, j]
                    thti =initwc[k, i, j]
                    eps = em[k, i, j]

                    # Set the boundname for the land surface cells
                    bndnm = "sage"

                    uz = [
                        iuzno,
                        (k, i, j),
                        lflag,
                        ivertcon,
                        surfdep,
                        vks,
                        thtr,
                        thtsx,
                        thti,
                        eps,
                        bndnm,
                    ]
                    uzf_packagedata.append(uz)

                    iuzno += 1

    ha = 0.2
    hroot = 0
    rootact = 0.0

    # Next prepare the stress period data for UZF
    # Store the steady state uzf stresses in dictionary
    uzf_perioddata = {}
    for t in range(num_ts):
        iuzno = 0
        spdx = []
        for i in range(0, iuzbnd.shape[0]):
            for j in range(0, iuzbnd.shape[1]):
                if iuzbnd[i, j] != 0:
                    finf = finf_grad[i, j] * uz_ts[t] # change precipitaiton
                    pet = uz_et[t]  # change ET
                    extdp = 1.5
                    if regdep[i, j]<2.0:
                        if regdep[i, j] - 0.2>0:
                            extdp = regdep[i, j]*3/4
                        else:
                            extdp = regdep[i, j]*3/4


                    extwc = 0.025
                    spdx.append(
                        [iuzno, finf, pet, extdp, extwc, ha, hroot, rootact]
                    )
                    iuzno += 1
        uzf_perioddata.update({t: spdx})



    # Set up runoff connections, which relies on a helper function inside a
    # companion script
    #
    # Leverages a function that uses the top elevation array and SFR locations to
    # calculate an array that is the equivalent of the irunbnd array from the UZF1
    # package.  The MVR package will be used to establish these connection in MF6
    # since the IRUNBND functionality went away in the new MF6 framework.
    import sfr_uzf_mvr_support_funcs as sageBld
    irunbnd = sageBld.determine_runoff_conns_4mvr(
        dat_pth, top, iuzbnd, sfrcells2, nrow, ncol
    )



    ## Mover package

    iuzno = 0
    k = 0  # Hard-wire the layer no.
    first0ok = True
    static_mvrperioddata = []
    for i in range(0, iuzbnd.shape[0]):
        for j in range(0, iuzbnd.shape[1]):
            if irunbnd[i, j] > 0:  # This is a uzf -> sfr connection
                iuzno = iuzno_cell_dict.get((i, j))
                if iuzno or first0ok:
                    static_mvrperioddata.append(
                        ('',"UZF-1", iuzno,'', "SFR-1", irunbnd[i, j] - 1, "FACTOR", 1.0)
                    )

                drn_idx = drn_dict.get((i, j))
                if drn_idx:
                    static_mvrperioddata.append(
                        ('',"DRN-1", drn_idx, '',"SFR-1", irunbnd[i, j] - 1, "FACTOR", 1.0)
                    )
                    first0ok = False

    mvrspd = {0: static_mvrperioddata}
    mvrpack = [["UZF-1"], ["SFR-1"], ["DRN-1"]]
    maxpackages = len(mvrpack)
    maxmvr = 1000000  # Something arbitrarily high
    fileID = random.randint(0, 1000000000000)


    # Instantiate the MODFLOW 6 simulation
    #os.mkdir('Tdatafile' + str(4))
    sim_ws = 'Tdatafile' + str(5)
    sim = flopy.mf6.MFSimulation(
        sim_name=example_name,
        version="mf6",
        sim_ws=sim_ws,
        exe_name=mf6exe,
    )

    # Instantiating MODFLOW 6 time discretization
    tdis_rc = []
    for i in range(len(perlen)):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))
    flopy.mf6.ModflowTdis(
        sim, nper=nper, perioddata=tdis_rc, time_units=time_units
    )


    # Instantiating MODFLOW 6 groundwater flow model
    gwfname = example_name
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
        newtonoptions="newton",
        model_nam_file="{}.nam".format(gwfname),
    )
    nouter, ninner = 1000, 1000
    hclose, rclose, relax = 5e-2, 3e-2, 0.97

    imsgwf = flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        complexity="complex",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="dbd",
        linear_acceleration="BICGSTAB",
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.1,
        under_relaxation_gamma=0.2,
        under_relaxation_momentum=0.0001,
        backtracking_number=20,
        backtracking_tolerance=1.1,
        backtracking_reduction_factor=0.2,
        backtracking_residual_limit=100,
        inner_dvclose=hclose,
        rcloserecord="1000.0 strict",
        inner_maximum=ninner,
        relaxation_factor=relax,
        number_orthogonalizations=2,
        preconditioner_levels=8,
        preconditioner_drop_tolerance=0.001,
        filename="{}.ims".format(gwfname),
    )
    sim.register_ims_package(imsgwf, [gwf.name])

    zbot = [bot1,bot2,bot3]

    # Instantiating MODFLOW 6 discretization package
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=zbot,
        idomain=[idomain,idomain,idomain],
        filename="{}.dis".format(gwfname),
    )

    flopy.mf6.ModflowGwfic(
        gwf, strt=[strt,strt2,strt3], filename="{}.ic".format(gwfname)
    )

    # Instantiating MODFLOW 6 node-property flow package
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=False,
        alternative_cell_averaging="AMT-HMK",
        icelltype=icelltype,
        k=[10**K_layer1*10**ani,10**K_layer1/10**K_layer2*10**ani2,10**K_layer1/10**K_layer2/10**K_layer3],
        k33=[10**K_layer1,10**K_layer1/10**K_layer2,10**K_layer1/10**K_layer2/10**K_layer3],
        save_specific_discharge=False,
        filename="{}.npf".format(gwfname),
    )

    # Instantiate MODFLOW 6 storage package
    flopy.mf6.ModflowGwfsto(
        gwf,
        ss=[5e-6,1e-6,5e-7],
        sy=[sy,sy2,sy3],
        iconvert=[iconvert,iconvert,iconvert],
        steady_state= False,#{0: True},
        transient= True, #{1: True},
        filename="{}.sto".format(gwfname),
    )

    # Instantiating MODFLOW 6 output control package for flow model
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="{}.bud".format(gwfname),
        head_filerecord="{}.hds".format(gwfname),
        headprintrecord=[
            ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )


    maxbound = len(drn_spd)  # The total number
    spd = {0: drn_spd}
    flopy.mf6.ModflowGwfdrn(
        gwf,
        pname="DRN-1",
        auxiliary=["ddrn"],
        auxdepthname="ddrn",
        print_input=False,
        print_flows=False,
        maxbound=maxbound,
        mover=True,
        stress_period_data=spd,  # wel_spd established in the MVR setup
        boundnames=False,
        save_flows=True,
        filename="{}.drn".format(gwfname),
    )

    # Instantiating MODFLOW 6 streamflow routing package
    flopy.mf6.ModflowGwfsfr(
        gwf,
        print_stage=False,
        print_flows=False,
        budget_filerecord=gwfname + ".sfr.bud",
        save_flows=True,
        mover=True,
        pname="SFR-1",
        unit_conversion=86400.0,
        boundnames=True,
        nreaches=len(conns),
        packagedata=pkdat,
        connectiondata=conns,
        perioddata=None,
        filename="{}.sfr".format(gwfname),
    )

    # Instantiating MODFLOW 6 unsaturated zone flow package
    flopy.mf6.ModflowGwfuzf(
        gwf,
        nuzfcells=nuzfcells,
        boundnames=True,
        mover=True,
        ntrailwaves=15,
        nwavesets=200,
        print_flows=False,
        save_flows=True,
        simulate_et=True,
        linear_gwet=True,
        wc_filerecord='WaterContent',
        packagedata=uzf_packagedata,
        perioddata=uzf_perioddata,
        budget_filerecord="{}.uzf.bud".format(gwfname),
        pname="UZF-1",
        filename="{}.uzf".format(gwfname),
    )

    maxbound = len(ghd_spd)  # The total number
    spd = {0: ghd_spd}
    flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
        gwf,
        print_input=False,
        print_flows=False,
        maxbound=maxbound,
        stress_period_data=spd,  # wel_spd established in the MVR setup
        boundnames=False,
        save_flows=True,
        filename="{}.ghb".format(gwfname),
    )

    mvrpack = [['','UZF-1'], ['','SFR-1'], ['','DRN-1']]

    flopy.mf6.ModflowGwfmvr(
        gwf,
        pname="MVR-1",
        maxmvr=maxmvr,
        modelnames=None,
        print_flows=False,
        maxpackages=maxpackages,
        packages=mvrpack,
        perioddata=mvrspd,
        budget_filerecord=gwfname + ".mvr.bud",
        filename="{}.mvr".format(gwfname),
    )

    sim.write_simulation(silent=True)

    success, buff = sim.run_simulation(silent=False)
    print(success)
    if success:
        break;


mf6 = sim
sim_name = sim.name
gwf = mf6.get_model(list(mf6.model_names)[0])
modobj = gwf.output.budget()
hdsobj = gwf.output.head()
sfrobj = gwf.sfr.output.budget()
uzfobj = gwf.uzf.output.budget()
ckstpkper = modobj.get_kstpkper()

## define the start
T_start = 0#(366+365 + 365 + 365)*1
res_path = 'S5_res'


head_tol = []
for num in range(num_ts)[T_start:]:
    hd_tmp = hdsobj.get_data(kstpkper=ckstpkper[num])
    hd_tmp = np.where(hd_tmp == 1e30, np.nan, hd_tmp)
    head_tol.append(hd_tmp)
head_tol = np.array(head_tol)
np.save(os.path.join(res_path, 'Head'),head_tol)



head_tol1 = []
for num in range(num_ts):
    hd_tmp = hdsobj.get_data(kstpkper=ckstpkper[num])
    hd_tmp = np.where(hd_tmp == 1e30, np.nan, hd_tmp)
    head_tol1.append(hd_tmp)
head_tol1 = np.array(head_tol1)


outflow = []

for kstpkper in ckstpkper[T_start:]:
    outletQ = sfrobj.get_data(
        kstpkper=kstpkper, text="    FLOW-JA-FACE"
    )
    outflow.append(outletQ[0][-1][2])

np.save(os.path.join(res_path,'Outflow'),outflow)


outflow1 = []

for kstpkper in ckstpkper:
    outletQ = sfrobj.get_data(
        kstpkper=kstpkper, text="    FLOW-JA-FACE"
    )
    outflow1.append(outletQ[0][-1][2])
    
    
Norm_unit = np.sum(idomain1)*5*5/1e3
 
def layer_org(cbb,df_GWbdg,name,num):
    STO_SS = cbb.get_data(text=name)[num]
    STO_SS_temp = STO_SS.reshape(3,-1)
    STO_SS0 = STO_SS_temp[0]
    STO_SS1 = STO_SS_temp[1]
    STO_SS2 = STO_SS_temp[2]

    df_GWbdg.append(np.sum(STO_SS0[STO_SS0>0]))
    df_GWbdg.append(np.sum(STO_SS0[STO_SS0<0]))
    
    df_GWbdg.append(np.sum(STO_SS1[STO_SS1>0]))
    df_GWbdg.append(np.sum(STO_SS1[STO_SS1<0]))
    
    df_GWbdg.append(np.sum(STO_SS2[STO_SS2>0]))
    df_GWbdg.append(np.sum(STO_SS2[STO_SS2<0]))
    return df_GWbdg

def uzf_layer_org(cbb,df_UZbdg,name,num):
    drn_tmp = cbb.get_data(text=name)[num]
    
    gwet_arr = np.zeros((3,top.shape[0],top.shape[1]))

    for itm in drn_tmp:
        i, j = iuzno_dict_rev[(itm[1] - 1)%np.sum(idomain1)]
        layer = int(np.floor((itm[1]-1)/np.sum(idomain1)))
        gwet_arr[layer,i, j] = itm[2]
    
    STO_SS0 = gwet_arr[0]
    STO_SS1 = gwet_arr[1]
    STO_SS2 = gwet_arr[2] 
    
    df_UZbdg.append(np.sum(STO_SS0[STO_SS0>0]))
    df_UZbdg.append(np.sum(STO_SS0[STO_SS0<0]))
    df_UZbdg.append(np.sum(STO_SS1[STO_SS1>0]))
    df_UZbdg.append(np.sum(STO_SS1[STO_SS1<0]))
    df_UZbdg.append(np.sum(STO_SS2[STO_SS2>0]))
    df_UZbdg.append(np.sum(STO_SS2[STO_SS2<0]))    
    
    return df_UZbdg

def og_data(cbb,name,num):
    drn_tmp = cbb.get_data(text=name)[num]
    drn_2 = []

    for itm in drn_tmp:
        drn_2.append(itm[2])
    drn_2 = np.array(drn_2)
    return drn_2    
    
 
# read the cell budget file
fname = os.path.join(sim_ws, "{}.bud".format(gwfname))
cbb = flopy.utils.CellBudgetFile(fname, precision="double")
df_GWbdg_tot = []

for num in range(num_ts)[T_start:]:

    df_GWbdg = []

    df_GWbdg = layer_org(cbb,df_GWbdg,name="STO-SS",num = num)

    df_GWbdg = layer_org(cbb,df_GWbdg,name="STO-SY",num = num)

    #'             WEl'
    drn_tmp = cbb.get_data(text="GHB")[num]
    drn_2 = []

    for itm in drn_tmp:
        drn_2.append(itm[2])
    drn_2 = np.array(drn_2)
    df_GWbdg.append(np.sum(drn_2[drn_2>0]))
    df_GWbdg.append(np.sum(drn_2[drn_2<0]))


    #'        UZF-GWET'

#     UZF_GWET_tmp = cbb.get_data(text=" UZF-GWET")[num]
#     UZF_GWET = []
#     for x, itm in enumerate(UZF_GWET_tmp):
#         UZF_GWET.append(itm[2])
#     UZF_GWET = np.array(UZF_GWET)

#     df_GWbdg.append(np.sum(UZF_GWET[UZF_GWET>0]))
#     df_GWbdg.append(np.sum(UZF_GWET[UZF_GWET<0]))
    df_GWbdg = uzf_layer_org(cbb,df_GWbdg," UZF-GWET",num)
    
        #'             DRN'
    drn_tmp = cbb.get_data(text="DRN")[num]
    drn_arr = []
    for itm in drn_tmp:
        drn_arr.append(itm[2])
    drn_arr = np.array(drn_arr)
    df_GWbdg.append(np.sum(drn_arr[drn_arr>0]))
    df_GWbdg.append(np.sum(drn_arr[drn_arr<0]))

        #'             DRN'
    drn_tmp = cbb.get_data(text="DRN-TO-MVR")[num]
    drn_arr = []
    for itm in drn_tmp:
        drn_arr.append(itm[2])
    drn_arr = np.array(drn_arr)
    df_GWbdg.append(np.sum(drn_arr[drn_arr>0]))
    df_GWbdg.append(np.sum(drn_arr[drn_arr<0]))
    
    
    df_GWbdg_tot.append(np.array(df_GWbdg))
    
    

df_GWD = pd.DataFrame(df_GWbdg_tot,
                   columns=['STO-SS_In0', 'STO-SS_Out0','STO-SS_In1', 'STO-SS_Out1', 'STO-SS_In2', 'STO-SS_Out2',                            
                            'STO-SY_In0', 'STO-SY_Out0','STO-SY_In1', 'STO-SY_Out1', 'STO-SY_In2', 'STO-SY_Out2',
                            'WEl_In','WEl_Out',
                            'UZF-GWET_In0','UZF-GWET_Out0','UZF-GWET_In1','UZF-GWET_Out1','UZF-GWET_In2','UZF-GWET_Out2',
                           'DRN_In','DRN_Out','DRN-TO-MVR_In','DRN-TO-MVR_Out'])
df_GWD.to_csv(os.path.join(res_path,'GWD_Bud.csv'))


fname = os.path.join(sim_ws, "{}.uzf.bud".format(gwfname))
cbb = flopy.utils.CellBudgetFile(fname, precision="double")

df_UZbdg_tot = []
for num in range(num_ts)[T_start:]:
    df_UZbdg = []

    df_UZbdg = uzf_layer_org(cbb,df_UZbdg,"STORAGE",num)
    df_UZbdg = uzf_layer_org(cbb,df_UZbdg,"INFILTRATION",num)
    df_UZbdg = uzf_layer_org(cbb,df_UZbdg,"REJ-INF",num)
    df_UZbdg = uzf_layer_org(cbb,df_UZbdg,"UZET",num)
    df_UZbdg = uzf_layer_org(cbb,df_UZbdg,"REJ-INF-TO-MVR",num)
    df_UZbdg_tot.append(np.array(df_UZbdg))
    
    
    
df_UZ = pd.DataFrame(df_UZbdg_tot,
                   columns=['STORAGE_In0', 'STORAGE_Out0','STORAGE_In1', 'STORAGE_Out1','STORAGE_In2', 'STORAGE_Out2',
                            'INFILTRATION_In0','INFILTRATION_Out0','INFILTRATION_In1','INFILTRATION_Out1','INFILTRATION_In2','INFILTRATION_Out2',
                            'REJ-INF_In0','REJ-INF_Out0','REJ-INF_In1','REJ-INF_Out1','REJ-INF_In2','REJ-INF_Out2',
                            'UZET_In0','UZET_Out0','UZET_In1','UZET_Out1','UZET_In2','UZET_Out2',
                           'REJ-INF-TO-MVR_In0','REJ-INF-TO-MVR_Out0','REJ-INF-TO-MVR_In1','REJ-INF-TO-MVR_Out1',
                            'REJ-INF-TO-MVR_In2','REJ-INF-TO-MVR_Out2'])
df_UZ.to_csv(os.path.join(res_path,'UZ_Bud.csv'))




fname = os.path.join(sim_ws, "{}.sfr.bud".format(gwfname))
cbb = flopy.utils.CellBudgetFile(fname, precision="double")
df_SFRbdg_tot = []

for num in range(num_ts)[T_start:]:
    df_SFRbdg = []
    REJ_INF_TO_MVR = og_data(cbb,"EXT-OUTFLOW",num)
    df_SFRbdg.append(np.sum(REJ_INF_TO_MVR[REJ_INF_TO_MVR>0]))
    df_SFRbdg.append(np.sum(REJ_INF_TO_MVR[REJ_INF_TO_MVR<0]))
    
    REJ_INF_TO_MVR = og_data(cbb,"GWF",num)
    df_SFRbdg.append(np.sum(REJ_INF_TO_MVR[REJ_INF_TO_MVR>0]))
    df_SFRbdg.append(np.sum(REJ_INF_TO_MVR[REJ_INF_TO_MVR<0]))
    
    REJ_INF_TO_MVR = og_data(cbb,"FROM-MVR",num)
    df_SFRbdg.append(np.sum(REJ_INF_TO_MVR[REJ_INF_TO_MVR>0]))
    df_SFRbdg.append(np.sum(REJ_INF_TO_MVR[REJ_INF_TO_MVR<0]))    
    df_SFRbdg_tot.append(np.array(df_SFRbdg))
    
df_SFR = pd.DataFrame(df_SFRbdg_tot,
                   columns=['EXT-OUTFLOW_In','EXT-OUTFLOW_Out',
                           'GWF_In','GWF_Out',
                           'FROM-MVR_In','FROM-MVR_Out'])  
df_SFR.to_csv(os.path.join(res_path,'SFR_Bud.csv'))

def binaryread(file, vartype, shape=(1,), charlen=16):
    """
    Uses numpy to read from binary file.  This was found to be faster than the
        struct approach and is used as the default.

    """

    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1)
    else:
        # find the number of values
        nval = np.prod(shape)
        result = np.fromfile(file, vartype, nval)
        if nval == 1:
            result = result  # [0]
        else:
            result = np.reshape(result, shape)
    return result
    
path = sim_ws
fpth = os.path.join(path , "WaterContent")
file = open(fpth,"rb")
WC_tot = []

for num in range(num_ts)[T_start:]:
    vartype = [
        ("kstp", "<i4"),
        ("kper", "<i4"),
        ("pertim", "<f8"),
        ("totim", "<f8"),
        ("text", "S16"),
        ("maxbound", "<i4"),
        ("1", "<i4"),
        ("11", "<i4"),
    ]
    #print(binaryread(file,vartype))
    binaryread(file,vartype)
    vartype = [
        ("data", "<f8"),
    ]

    WC_arr = np.zeros((nlay,top.shape[0],top.shape[1]))*np.nan
    
    
    for k in range(nlay):
        for n in range(int(nuzfcells/nlay)):
        
            i, j = iuzno_dict_rev[n]
            WC_arr[k, i, j] = np.array(binaryread(file,vartype).tolist())

    WC_tot.append(WC_arr)

file.close()



np.save(os.path.join(res_path, 'WC'),WC_tot)



num = 0
df_gwd = pd.read_csv(os.path.join(res_path,'GWD_Bud.csv'))
df_uz = pd.read_csv(os.path.join(res_path,'UZ_Bud.csv'))
df_sfr = pd.read_csv(os.path.join(res_path,'SFR_Bud.csv'))

storage_change0 = abs(df_gwd['STO-SS_Out0'][num:]) - abs(df_gwd['STO-SS_In0'][num:]) + abs(df_gwd['STO-SY_Out0'][num:]) - \
                  abs(df_gwd['STO-SY_In0'][num:]) + abs(df_uz['STORAGE_Out0'][num:]) - abs(df_uz['STORAGE_In0'][num:])   

storage_change1 = abs(df_gwd['STO-SS_Out1'][num:]) - abs(df_gwd['STO-SS_In1'][num:]) + abs(df_gwd['STO-SY_Out1'][num:]) - \
                  abs(df_gwd['STO-SY_In1'][num:]) + abs(df_uz['STORAGE_Out1'][num:]) - abs(df_uz['STORAGE_In1'][num:]) 

storage_change2 = abs(df_gwd['STO-SS_Out2'][num:]) - abs(df_gwd['STO-SS_In2'][num:]) + abs(df_gwd['STO-SY_Out2'][num:]) - \
                  abs(df_gwd['STO-SY_In2'][num:]) + abs(df_uz['STORAGE_Out2'][num:]) - abs(df_uz['STORAGE_In2'][num:]) 

storage_change0 = storage_change0/Norm_unit
storage_change1 = storage_change1/Norm_unit
storage_change2 = storage_change2/Norm_unit 

outflow_sum = abs(df_sfr['EXT-OUTFLOW_Out'][num:])

outflow_sum = outflow_sum/Norm_unit  

ETsum0 =  abs(df_gwd['UZF-GWET_Out0'][num:]) + abs(df_uz['UZET_Out0'][num:])
ETsum1 =  abs(df_gwd['UZF-GWET_Out1'][num:]) + abs(df_uz['UZET_Out1'][num:])
ETsum2 =  abs(df_gwd['UZF-GWET_Out2'][num:]) + abs(df_uz['UZET_Out2'][num:])

ETsum0 = ETsum0/Norm_unit 

REJ0 = abs(df_uz['REJ-INF_Out0'][num:]) - abs(df_uz['REJ-INF_In0'][num:]) + abs(df_gwd['DRN_Out'][num:])
REJ1 = abs(df_uz['REJ-INF_Out1'][num:]) - abs(df_uz['REJ-INF_In1'][num:])
REJ2 = abs(df_uz['REJ-INF_Out2'][num:]) - abs(df_uz['REJ-INF_In2'][num:])

REJ0 = REJ0/Norm_unit 

Wel = abs(df_gwd['WEl_Out'][num:]) - abs(df_gwd['WEl_In'][num:])

Wel = Wel/Norm_unit 


input_inf0 = abs(df_uz['INFILTRATION_Out0'][num:]) - abs(df_uz['INFILTRATION_In0'][num:])
input_inf1 = abs(df_uz['INFILTRATION_Out1'][num:]) - abs(df_uz['INFILTRATION_In1'][num:])
input_inf2 = abs(df_uz['INFILTRATION_Out2'][num:]) - abs(df_uz['INFILTRATION_In2'][num:])

input_inf0 = input_inf0/Norm_unit 


temp = (uz_et[:num_ts]*1e3)- (ETsum0 + REJ0)
temp[temp>=0]=0

outflow_sum = outflow_sum + abs(temp)
ETsum0 = ETsum0 + REJ0 - abs(temp)


np.save(os.path.join(res_path, 'outflow_sum'),outflow_sum)
np.save(os.path.join(res_path, 'ETsum0'),ETsum0)
np.save(os.path.join(res_path, 'Wel'),Wel)
np.save(os.path.join(res_path, 'storage_change0'),storage_change0)
np.save(os.path.join(res_path, 'storage_change1'),storage_change1)
np.save(os.path.join(res_path, 'storage_change2'),storage_change2)
np.save(os.path.join(res_path, 'input_inf0'),input_inf0)
















