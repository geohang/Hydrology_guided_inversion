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




K_layer1 = 10.0 #10**data['parK_layer1'][index]*0.8
K_layer2 = 1 #10**data['parK_layer1'][index]/10**data['parK_layer2'][index]
K_layer3 = 0.001 #10**data['parK_layer1'][index]/10**data['parK_layer2'][index]/10**data['parK_layer3'][index]
porosity1 = 0.45#data['parporosity1'][index]
porosity2 = 0.35#data['parporosity2'][index]
porosity3 = 0.05#data['parporosity3'][index]
rate = -4.0 #data['parrate'][index]
ani = np.log(1)#data['parani'][index] 
ani2 = np.log(1)

L1_grid = 4
#K_L1 = ( K_layer1*0.9 - K_layer1*1.1 )/(L1_grid)*np.arange(L1_grid) + K_layer1*1.1
#P_L1 = ( porosity1*0.9 - porosity1*1.1 )/(L1_grid)*np.arange(L1_grid) + porosity1*1.1

L2_grid = 10
#K_L2 = ( K_layer2*0.8 - K_layer2*1.3 )/(L2_grid)*np.arange(L2_grid) + K_layer2*1.3
#P_L2 = ( porosity2*0.9 - porosity2*1.1 )/(L2_grid)*np.arange(L2_grid) + porosity2*1.1
rho_w = 1000
g = 9.8
d_g = 1e-4
mu_w = 1e-3

#K_L1 = 10**( (np.log10(K_layer2) - np.log10(K_layer1) )/(L1_grid-1)*np.arange(L1_grid) + np.log10(K_layer1) )
P_L1 = ( porosity2 - porosity1 )/(L1_grid-1)*np.arange(L1_grid) + porosity1
K_L1 = rho_w*g*d_g**2*P_L1**3/(mu_w*180*(1-P_L1)**2)*60*60*24

#K_L2 = 10**( (np.log10(K_layer3) - np.log10(K_layer2))/(L2_grid-1)*np.arange(L2_grid) + np.log10(K_layer2) )
P_L2 = ( porosity3 - porosity2 )/(L2_grid-1)*np.arange(L2_grid) + porosity2
K_L2 = rho_w*g*d_g**2*P_L2**3/(mu_w*180*(1-P_L2)**2)*60*60*24




print(K_L1)
print(K_L2)

config.base_ws = './Field1'

ws = config.base_ws
example_name = "TLnewtest2sfb"
length_units = "meters"
time_units = "days"
nlay = L1_grid + L2_grid   # Number of layers in parent model

coeff1 = 8
coeff2 = 8.5
coeff3 = 5.0

# Number of columns in parent model
delr = 1  # Parent model column width ($m$)
delc = 1 # Parent model row width ($m$

# Time related variables
num_ts = 365+365+365+365+365+365   #(365 + 366 + 365 +365)*1+366 + 365  #365 + 365 + 365 + 366 + 366 #(365 + 366 + 365)*3 + 365 + 366 
perlen = [1]*num_ts
nper = len(perlen)
nstp = [1] * num_ts
tsmult = [1.0] * num_ts


# from mf-nwt .dis file
dat_pth = example_name
top = np.loadtxt(os.path.join(dat_pth, "top.txt"))
regdep = np.loadtxt(os.path.join(dat_pth, "regdep_new1m.txt"))
fradep = np.loadtxt(os.path.join(dat_pth, "fradep_new1m.txt"))
idomain1 = np.loadtxt(os.path.join(dat_pth, "id.txt"))

regdep = np.loadtxt(os.path.join(dat_pth, 'reg_depstr.txt'))#np.loadtxt('reg_depstr.txt')
           
fradep =  regdep + 40


bot1 = top - regdep
bot2 = top - fradep

dis =  bot2.copy()
dis[idomain1 == 0] = np.nan
bot3 = dis - 30


bot = []
for i in range(L1_grid):
    bot.append(top - 1/(L1_grid)*regdep*(i+1))
    
for i in range(L2_grid):
    bot.append(bot1 - 1/(L2_grid)*(bot1 - bot2)*(i+1))
#bot.append(bot3)    

for JJJ in range(1500)[0:]:


     strt = []
     IDD = []
     for i in range(nlay):
         strt.append(bot[nlay-9] - float(JJJ)/100)
         IDD.append(idomain1)
         
     print(float(JJJ)/100)    
     # from mf-nwt .bas file
     
     nrow = top.shape[0]  # Number of rows in parent model
     ncol = top.shape[1]
     
     idomain = np.abs(idomain1)
     
     # from mf-nwt .upw file
     theta_res = 0.025
     sy = []
     
     for i in range(L1_grid):
         sy.append((P_L1[i]-theta_res))
         
     for i in range(L2_grid):
         sy.append((P_L2[i]-theta_res))
         
     #sy.append(porosity3-theta_res)    
     
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
     
     
     rhk = 1 #K_layer1
     man = 0.035
     ustrf = 1.0
     ndv = 0
     pkdat = []
     sfrcells = []
     sfrcells2 = []
     for i in np.arange(len(df)):
     
         pkdat.append(
                 (
                 i, # rho
                 (df[1][i]-1,df[2][i]-1,df[3][i]-1), # k, i ,j
                 df[4][i], # rlen
                 df[5][i], # rwid
                 df[6][i], # rgrd
                 df[7][i], # rtp
                 df[8][i],# rbth
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
                 ghd_spd.append([(nlay-9, i, j),bot[nlay-9][i,j] ,cond])  #  'ddrn',
                 # append dictionary of drain indices
                 ighdno += 1
                 
     # Prepping input for UZF package
     # Package_data information
     
     iuzbnd = idomain.copy()
     
     thts = []
     for i in range(L1_grid):
         thts.append(np.ones((top.shape))*(P_L1[i]))    
     for i in range(L2_grid):
         thts.append(np.ones((top.shape))*(P_L2[i]))
     #thts.append(np.ones((top.shape))*porosity3)   
     thts = np.array(thts)
     
     uzk33 = []
     for i in range(L1_grid):
         uzk33.append(np.ones((top.shape))*(K_L1[i]))
         
     for i in range(L2_grid):
         uzk33.append(np.ones((top.shape))*(K_L2[i]))
     #uzk33.append(np.ones((top.shape))*K_layer3)  
     uzk33 = np.array(uzk33)
     
     initwc = []
     for i in range(L1_grid):
         initwc.append(np.ones((top.shape))*(P_L1[i]*0.7))
         
     for i in range(L2_grid):
         initwc.append(np.ones((top.shape))*(P_L2[i]*0.7))
     #initwc.append(np.ones((top.shape))*theta_res) 
     initwc = np.array(initwc)
     
     em = []
     for i in range(L1_grid):
         em.append(np.ones((top.shape))*coeff1)
         
     for i in range(L2_grid):
         em.append(np.ones((top.shape))*coeff2)
     #em.append(np.ones((top.shape))*coeff3)
     em = np.array(em)
     
     
     finf_grad = np.ones((top.shape))
     # next, load time series of multipliers
     uz_ts = np.loadtxt(os.path.join(dat_pth, "TLoutflow_11_15.txt"))
     uz_ts = np.hstack((uz_ts[:365],uz_ts[:365],
                        uz_ts[:365],uz_ts[:365],uz_ts[:365],uz_ts[:365],uz_ts[:365]))
     
     uz_ts = uz_ts*2.0
     uz_et = np.loadtxt(os.path.join(dat_pth, "TLET_11_15.txt"))
     uz_et = np.hstack((uz_et[:365],uz_et[:365],
                        uz_et[:365],uz_et[:365],uz_et[:365],uz_et[:365],uz_et[:365]))
     
     uzf_packagedata = []
     pd0 = []
     iuzno_cell_dict = {}
     iuzno_dict_rev = {}
     iuzno = 0
     surfdep = 0.15/20
     
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
     sim_ws = sim_ws = '/bsuhome/hangchen/scratch/Test5/'
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
     nouter, ninner = 2000, 2000
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
         under_relaxation_kappa=0.08,
         under_relaxation_gamma=0.05,
         under_relaxation_momentum=0.0,
         backtracking_number=20,
         backtracking_tolerance=2.0,
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
     
     K_totx = []
     for i in range(L1_grid):
         K_totx.append(K_L1[i]*10**ani)
         
     for i in range(L2_grid):
         K_totx.append(K_L2[i]*10**ani2)
     #K_totx.append(K_layer3)
     K_totx = np.array(K_totx)
     
     K_totz = []
     for i in range(L1_grid):
         K_totz.append(K_L1[i])
         
     for i in range(L2_grid):
         K_totz.append(K_L2[i])
     #K_totz.append(K_layer3)
     K_totz = np.array(K_totz)
     
     
     ss = []
     for i in range(L1_grid):
         ss.append(5e-6)    
     for i in range(L2_grid):
         ss.append(1e-6)
     
     #ss.append(5e-7)
     
     # Instantiating MODFLOW 6 discretization package
     flopy.mf6.ModflowGwfdis(
         gwf,
         nlay=nlay,
         nrow=nrow,
         ncol=ncol,
         delr=delr,
         delc=delc,
         top=top,
         botm=bot,
         idomain=IDD,
         filename="{}.dis".format(gwfname),
     )
     
     flopy.mf6.ModflowGwfic(
         gwf, strt=strt, filename="{}.ic".format(gwfname)
     )
     
     # Instantiating MODFLOW 6 node-property flow package
     flopy.mf6.ModflowGwfnpf(
         gwf,
         save_flows=False,
         alternative_cell_averaging="AMT-HMK",
         icelltype=icelltype,
         k = K_totx,
         k33 = K_totz,
         save_specific_discharge=False,
         filename="{}.npf".format(gwfname),
     )
     
     
     # Instantiate MODFLOW 6 storage package
     flopy.mf6.ModflowGwfsto(
         gwf,
         ss=ss,
         sy=sy,
         iconvert=iconvert,
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
         nwavesets=300,
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
     
     success, buff = sim.run_simulation(silent=True)
     if success:
        break;

def og_data(cbb,name,num):
    drn_tmp = cbb.get_data(text=name)[num]
    drn_2 = []

    for itm in drn_tmp:
        drn_2.append(itm[2])
    drn_2 = np.array(drn_2)
    return drn_2   
    
T_start = 365*3 #(365 + 366 + 365 +365)*1 
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
WC_tot = np.array(WC_tot)

mf6 = sim
sim_name = sim.name
gwf = mf6.get_model(list(mf6.model_names)[0])
modobj = gwf.output.budget()
hdsobj = gwf.output.head()
sfrobj = gwf.sfr.output.budget()
uzfobj = gwf.uzf.output.budget()
ckstpkper = modobj.get_kstpkper()


outflow = []
df_GWbdg = []
rej = []
uz_ET = []
gw_ET = []

for kstpkper in ckstpkper:

   # 9. Get flows at outlet
   outletQ = sfrobj.get_data(
       kstpkper=kstpkper, text="    FLOW-JA-FACE"
   )
   outflow.append(outletQ[0][-1][2])
   
   drn_tmp = modobj.get_data(text="DRN",kstpkper=kstpkper)
   drn_arr = []
   for itm in drn_tmp[0]:
       drn_arr.append(itm[2])
   drn_arr = np.array(drn_arr)
   df_GWbdg.append(np.sum(drn_arr[drn_arr<0]))

   drn_tmp = modobj.get_data(text="UZF-GWET", kstpkper=kstpkper)
   drn_arr = []
   for itm in drn_tmp[0]:
       drn_arr.append(itm[2])
   drn_arr = np.array(drn_arr)
   gw_ET.append(np.sum(drn_arr[drn_arr<0]))


   drn_tmp = uzfobj.get_data(text="REJ-INF",kstpkper=kstpkper)    
   tt = []
   for itm in drn_tmp[0]:
       tt.append(itm[2]) 
   tt = np.array(tt)
   rej.append(np.sum(tt[tt<0]))
   
   
   drn_tmp = uzfobj.get_data(text="UZET",kstpkper=kstpkper)    
   tt = []
   for itm in drn_tmp[0]:
       tt.append(itm[2]) 
   tt = np.array(tt)
   uz_ET.append(np.sum(tt[tt<0]))


Norm_unit = 16.319
outflow = np.array(outflow)
df_GWbdg = abs(np.array(df_GWbdg))
rej = abs(np.array(rej))
gw_ET = abs(np.array(gw_ET))
uz_ET= abs(np.array(uz_ET))

temp = (uz_et[:len(outflow)]*1e3)- (gw_ET + uz_ET + rej + df_GWbdg)/Norm_unit
temp[temp>=0]=0     

outflow = outflow/Norm_unit + abs(temp)


#outflow = []
#
#for kstpkper in ckstpkper[T_start:]:
#    outletQ = sfrobj.get_data(
#        kstpkper=kstpkper, text="    FLOW-JA-FACE"
#    )
#    outflow.append(outletQ[0][-1][2])

np.save('Outflow',outflow)

head_tol = []
for num in range(num_ts)[T_start:]:
    hd_tmp = hdsobj.get_data(kstpkper=ckstpkper[num])
    hd_tmp = np.where(hd_tmp == 1e30, np.nan, hd_tmp)
    head_tol.append(hd_tmp)
head_tol = np.array(head_tol)

P_all = np.hstack((np.concatenate((P_L1,P_L2)), porosity3))
plt.plot(P_all)


def returnWC(top,bot,head_tol,WC_tot,P_all):

    wc_temp = np.zeros(bot.shape)*np.nan

    gw = head_tol - bot
    flags1 = gw <=0
    thickness = top - bot
    uz = top - head_tol
    flags2 = uz<=0

    wc_temp[flags1] = WC_tot[flags1]
    wc_temp[flags2] = P_all

    flags3 = np.logical_and(np.logical_not(flags2),np.logical_not(flags1))
    wc_temp[flags3] = (uz[flags3]*WC_tot[flags3] + gw[flags3]*P_all )/thickness[flags3]
    
    return wc_temp
    
WC_tot_new = WC_tot.copy()

for i in range(num_ts):
    WC_tot_new[i,0] = returnWC(top,bot[0],head_tol[i,0],WC_tot[i,0],P_all[0])
    for j in range(14)[1:]:
        WC_tot_new[i,j] = returnWC(bot[j-1],bot[j],head_tol[i,j],WC_tot[i,j],P_all[j])
        

np.save('Watercontent',WC_tot_new)
    
np.save('uz_ts',uz_ts[:365 + 366])
np.save('uz_et',uz_et[:365 + 366])




