#Importar librerias
import numpy as np
import MDAnalysis as mda
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from MDAnalysis.analysis import rms, align, distances
from MDAnalysis.analysis.rms import RMSF
from pydmd import DMD, BOPDMD, EDMD
from pydmd.plotter import plot_eigs, plot_summary
import os

def Cont_Res(u,atomgroup1, atomgroup2,cutoff=14):

    n_rbd = len(atomgroup1.residues)  #atomgroup1 protein
    n_ace2 = len(atomgroup2.residues) #atomgroup2 oxygen of the surface
    counter=np.zeros((n_rbd,n_ace2))

    for ts in u.trajectory:
        if ts.frame==0:
            continue
        res_coms1= np.array([r.atoms.center_of_mass() for r in atomgroup1.residues])
        res_coms2= atomgroup2.atoms.positions
        dist_arr2= distances.distance_array(res_coms1, res_coms2, box=u.dimensions)
        mask= (dist_arr2 <cutoff)*1
        counter+=mask#np.tril(mask, -1)

    sum_count=np.sum(counter,axis=1)
    return atomgroup1.residues.resids,sum_count,atomgroup1.residues.resnames#,counter

fold='omicron_open_10'
topology='npt.gro'
trajectory='md_short_compact.xtc'

u = mda.Universe(os.path.join(fold,topology),os.path.join(fold,trajectory))
timesteps=len(u.trajectory)
cutoff = 14
prom = 0.75
r=10 #number of modes retained

atomgroup1= u.select_atoms('not(resname SOL or resname DOL or resname NA or resname CL) and prop z<120') #solo proteina
atomgroup2= u.select_atoms('resname DOL and name O1 and prop z > 16') #solo superficie con la parte de oxigenos superior\
hist1,hist2,names=Cont_Res(u,atomgroup1, atomgroup2,cutoff=cutoff)

df = pd.DataFrame({'names': names,'# residuo': hist1,'conteo': hist2})
df_g1 = df[df['names'].str.len() == 3]
df_glycan = df[df['names'].str.len() == 4]
prom_min_g1 = (df_g1['conteo'].max())*prom
prom_min_glycan = (df_glycan['conteo'].max())*prom
df_g1_filtered = df_g1[df_g1['conteo'] >= prom_min_g1]
df_glycan_filtered = df_glycan[df_glycan['conteo'] >= prom_min_glycan]

str_g1 = ' '.join(df_g1_filtered['# residuo'].astype(str)) # Convert column residue to string and concatenate values
str_glycan = ' '.join(df_glycan_filtered['# residuo'].astype(str)) # Convert column residue to string and concatenate values
selection_protein = 'resid '+str_g1
selection_glycan = 'resid '+str_glycan

domain_group1 = u.select_atoms(selection_protein) #resid residue-number-range (inclusive)
domain_glycan = u.select_atoms(selection_glycan)
domain=domain_glycan + domain_group1
nt = timesteps
X1 = np.zeros((len(domain)*3,nt)) #Trajectory matrix
for nt_step in u.trajectory:
    X1[:,nt_step.frame] = np.reshape(domain.positions,-1) # [x1,y1,z1,x2,y2,z2,...] can be improved!
t = np.linspace(1,nt,nt)
optdmd = BOPDMD(svd_rank=r)#, varpro_opts_dict={"verbose": True, "tol": 0.04}) #uncomment to add options
optdmd.fit(X1, t)
 
n_atoms = len(domain) #create .xtc for dmd reconstructed trajectory
X1_reshaped = np.real(optdmd.reconstructed_data).T.reshape(nt, n_atoms, 3)  # Reshape X1 to [n_timesteps, n_atoms, 3]

rmsd_dmd = {}
rmsf_dmd = {}
rog_dmd = {'group1':[],'glycan':[],'all':[]}
rmsd = {}
rmsf = {}
rog = {'group1':[],'glycan':[],'all':[]}
dict_domain = {'group1':selection_protein, 
               'glycan':selection_glycan,
               'all': f'{selection_protein} or {selection_glycan}'}

for key,value in dict_domain.items():
    domain_dmd = mda.Merge(domain).load_new(
            X1_reshaped[:, :, :], order="fac") #frame atom coordinate
    #RMSD
    aligner = align.AlignTraj(domain_dmd, domain_dmd, select=value, in_memory=True)
    aligner.run()
    R = mda.analysis.rms.RMSD(domain_dmd, domain_dmd, select=value, ref_frame=0)              
    R.run()
    rmsd_dmd[key] = R.rmsd.T

    #RMSF
    selection = domain_dmd.select_atoms(value)
    reference_coordinates = domain_dmd.trajectory.timeseries(asel=selection).mean(axis=1)
    reference = mda.Merge(selection).load_new(
                reference_coordinates[:, None, :], order="afc")   
    rmsf_dmd[key] = RMSF(selection, verbose=True).run()
    
for key,value in dict_domain.items():
    u = mda.Universe(os.path.join(fold,topology),os.path.join(fold,trajectory))
    #RMSD
    aligner = align.AlignTraj(u, u, select=value, in_memory=True)
    aligner.run()
    R = mda.analysis.rms.RMSD(u, u, select=value, ref_frame=0)              
    R.run()
    rmsd[key] = R.rmsd.T 

    #RMSF
    selection = u.select_atoms(value)
    reference_coordinates = u.trajectory.timeseries(asel=selection).mean(axis=1)
    reference = mda.Merge(selection).load_new(
                reference_coordinates[:, None, :], order="afc")           
    aligner = align.AlignTraj(u, reference, select=value, in_memory=True).run()
    rmsf[key] = RMSF(selection, verbose=True).run()

for key,value in dict_domain.items(): #Based on MDAnalysis script
    group_dmd = domain_dmd.select_atoms(value)
    group = u.select_atoms(value)
    for ntime in domain_dmd.trajectory:
        rog_dmd[key].append(group_dmd.atoms.radius_of_gyration())
    for ntime in u.trajectory:
        rog[key].append(group.atoms.radius_of_gyration())
    rog_dmd[key] = np.array(rog_dmd[key])
    rog[key] = np.array(rog[key])

#Plot
fig = plt.figure(figsize=(8,5))
plt.plot(rmsd['glycan'][0]*u.coord.dt, rmsd_dmd['glycan'][2], 'k--',  label="glycan (dmd)",linewidth=4); 
plt.plot(rmsd['glycan'][1], rmsd['glycan'][2], 'k',  label="glycan (MD)");
plt.plot(rmsd['group1'][0]*u.coord.dt, rmsd_dmd['group1'][2], 'g--',  label="group1 (dmd)",linewidth=4); 
plt.plot(rmsd['group1'][1], rmsd['group1'][2], 'g',  label="group1 (MD)");
plt.legend(loc="best"); plt.xlabel("time (ps)"); plt.ylabel(r"RMSD ($\AA$)");
plt.savefig(f'afm_md_dmd/plots/RMSD_dmd_MD_{fold}.png')

fig, axs = plt.subplots(2, 2, figsize=(15,8))
axs[0,0].bar(range(len(domain_dmd.select_atoms(dict_domain['group1']))), rmsf_dmd['group1'].rmsf,color='b')
axs[0,1].bar(range(len(domain_dmd.select_atoms(dict_domain['glycan']))), rmsf_dmd['glycan'].rmsf,color='r')
axs[1,0].bar(range(len(domain.select_atoms(dict_domain['group1']))), rmsf['group1'].rmsf,color='b')
axs[1,1].bar(range(len(domain.select_atoms(dict_domain['glycan']))), rmsf['glycan'].rmsf,color='r')
axs[0, 0].set_title('Group 1 (DMD)'); axs[0, 1].set_title('Glycan (DMD)')
axs[1, 0].set_title('Group 1 (MD)'); axs[1, 1].set_title('Glycan (MD)')
fig.savefig(f'afm_md_dmd/plots/RMSF_dmd_MD_{fold}.png')

fig, axs = plt.subplots(1, 1, figsize=(15,8))
axs.plot(range(len(rog['group1'])), rog['group1'],color='b',label='group1')
axs.plot(range(len(rog['glycan'])), rog['glycan'],color='r',label='glycan')
axs.plot(range(len(rog['all'])), rog['all'],color='m',label='all')
axs.plot(range(len(rog_dmd['group1'])), rog_dmd['group1'],color='b',linestyle='--',linewidth=4)
axs.plot(range(len(rog_dmd['glycan'])), rog_dmd['glycan'],color='r',linestyle='--',linewidth=4)
axs.plot(range(len(rog_dmd['all'])), rog_dmd['all'],color='m',linestyle='--',linewidth=4)
axs.legend(); axs.set_title('MD (-) / DMD (--)'); 
fig.savefig(f'afm_md_dmd/plots/RoG_dmd_MD_{fold}.png')