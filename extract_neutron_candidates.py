print('\nLoading Packages...')
import os          
import numpy as np
import uproot
from tqdm import trange
from scipy.stats import norm
import re

# ----------------------------------------------------- #

# edit accordingly

data_directory = 'data/'                                     # directory containing BeamClusterAnalysis ntuples
waveform_dir = 'AmBe_waveforms/'                             # directory containing raw AmBe PMT waveforms

file_pattern = re.compile(r'R(\d+)_AmBe\.ntuple\.root')      # Pattern to extract run numbers from the files: R<run_number>_AmBe.ntuple.root -- edit to match your filename pattern

output_filename = 'neutron_candidates.root'                  # name of output root file to be created containing neutron candidates

which_Tree = 1                                               # PhaseIITreeMaker (0) or ANNIEEventTreeMaker (1) tool

# ----------------------------------------------------- #

# AmBe neutrons
def AmBe(CPE, CCB, CT, CN):
    if(CPE<=0 or CPE>70):      # 0 < cluster PE < 100
        return False
    if(CCB>=0.4 or CCB<=0):   # Cluster Charge Balance < 0.4
        return False
    if(CT<2000):              # cluster time not in prompt window
        return False
    if(CN!=1):                # only 1 neutron candidate cluster per trigger
        return False          # for the PhaseIITreeMaker, this MUST BE CHANGED --> "clusterNumber" just tags the current cluster. Need to make sure this is the ONLY cluster in the event
    return True

# cosmic muon clusters
def cosmic(CT, CPE):
    if(CT<2000):              # any cluster in the prompt (2us) window
        return True
    if(CPE>150):              # any cluster > 150 PE in the prompt or ext window
        return True
    return False

# grab source location based on the run number
def source_loc(run):
    
    run = int(run)
    
    source_positions = {
        
        # Port 5 data
        4506: (0, 100, 0),
        4505: (0, 50, 0),
        4499: (0, 0, 0),
        4507: (0, -50, 0),
        4508: (0, -100, 0),
        
        # Port 1 data
        4593: (0, 100, -75),
        4590: (0, 50, -75),
        4589: (0, 0, -75),
        4596: (0, -50, -75),
        4598: (0, -100, -75),
        
        # Port 4 data
        4656: (-75, 100, 0),
        4658: (-75, 50, 0),
        4660: (-75, 0, 0),
        4662: (-75, -50, 0), 4664: (-75, -50, 0), 4665: (-75, -50, 0), 4666: (-75, -50, 0), 4667: (-75, -50, 0), 4670: (-75, -50, 0),
        4678: (-75, -100, 0), 4683: (-75, -100, 0), 4687: (-75, -100, 0),
        
        # Port 3 data
        4628: (0, 100, 102), 4629: (0, 100, 102),
        4633: (0, 50, 102),
        4635: (0, 0, 102), 4636: (0, 0, 102), 4640: (0, 0, 102), 4646: (0, 0, 102),
        4649: (0, -50, 102),
        4651: (0, -100, 102),
        
    }
        
    if run in source_positions:
        return source_positions[run]
    
    print('\n##### RUN NUMBER ' + run + 'DOESNT HAVE A SOURCE LOCATION!!! ERROR #####\n')
    exit()

    
# ----------------------------------------------------- #

# Load Data

file_names = []; run_numbers = []

for file_name in os.listdir(data_directory):
    match = file_pattern.match(file_name)
    if match:
        # Extract the run number
        run_number = int(match.group(1))
        run_numbers.append(str(run_number))

        # Store the full file path
        file_names.append(os.path.join(data_directory, file_name))

# ----------------------------------------------------- #
# our analysis arrays

cluster_time = []; cluster_charge = []; cluster_QB = []; cluster_hits = []; 
hit_times = []; hit_charges = []; hit_ids = []
source_position = [[], [], []]      # x, y, z

# add any additional arrays you would like. Be sure to:
#        - create the array
#        - extract that information from the BeamCluster Trees accordingly
#        - create output file branch and populate it accordingly

# ----------------------------------------------------- #
folder_pattern = re.compile(r'^RWM_\d+')    # for the RWM waveform root files

# extracted based on analysis of AmBe PMT waveforms

pulse_start = 300      # adc
pulse_end = 1200       # adc

                       # pulse integrated charge cuts
pulse_pedestal = 120   # pedestal cut (not used in the script, here for documentation)
pulse_gamma = 400      # higher-energy peak (4.43 MeV gamma in BGO)
pulse_max = 1000       # high energy cutoff (arbitrary)

NS_PER_ADC_SAMPLE = 2
ADC_IMPEDANCE = 50     # Ohms


c1 = 0
for run in run_numbers:
    
    print('\n\nRun: ', run, '(', (c1+1), '/', len(file_names), ')')
    print('-----------------------------------------------------------------')
    
    # grab source location based on run number
    x_pos, y_pos, z_pos = source_loc(run)
    print('Source position (x,y,z): ' + ' (' + str(x_pos) + ',' + str(y_pos) + ',' + str(z_pos) + ')')

    good_events = []
    accepted_events = 0; rejected_events = 0
    counter = 0

    waveform_files = os.listdir(waveform_dir + run + '/')
    
    # First step, load the waveforms
    print('\nLoading Raw Waveforms...')
    for file in trange(len(waveform_files)):
        
        waveform_filepath = waveform_dir + run + '/' + waveform_files[file]
        
        with uproot.open(waveform_filepath) as root:
            folder_names = [name for name in root.keys() if folder_pattern.match(name)]

            for folder in folder_names:
                timestamp = folder.split('_')[-1].split(';')[0]

                try:
                    hist = root[folder]
                    hist_values = hist.values()
                    hist_edges = hist.axes[0].edges()

                    baseline, sigma = norm.fit(hist_values)
                    
                    # Calculate integrated charge using NumPy for efficiency
                    pulse_mask = (hist_edges[:-1] > pulse_start) & (hist_edges[:-1] < pulse_end)
                    IC = np.sum(hist_values[pulse_mask] - baseline)
                        
                    IC_adjusted = (NS_PER_ADC_SAMPLE / ADC_IMPEDANCE) * IC

                    # check if the pulse comes from the 4.43 MeV gamma
                    if pulse_max > IC_adjusted > pulse_gamma:
                        
                        post_pulse_mask = hist_edges[:-1] > pulse_end
                        post_pulse_values = hist_values[post_pulse_mask]
                        another_pulse = np.any(post_pulse_values > (7 + sigma + baseline))
                        
                        if not another_pulse:   # there were no other pulses
                            good_events.append(int(timestamp))
                            accepted_events += 1
                        else:
                            rejected_events += 1

                    else:    # the pulse is too small
                        rejected_events += 1

                except Exception as e:
                    print(f"Could not access '{folder}': {e}")

                counter += 1   # update event counter


    print('\nThere were a total of', (accepted_events + rejected_events), 'aquisitions')
    print(accepted_events, 'waveforms were accepted (', \
          round(100*accepted_events/(accepted_events+rejected_events),2), '%)')
    print(rejected_events, 'waveforms were rejected (', \
          round(100*rejected_events/(accepted_events+rejected_events),2), '%)\n')

    good_events = set(good_events)    # convert list to set to speed computation


    # Second step, load data
    print('\nLoading AmBe event data...')

    cosmic_events = 0; total_events = 0; neutron_cand_count = 0
    with uproot.open(file_names[c1]) as file_1:     # c1 iterated through each loop

        if which_Tree == 0:
            Event = file_1["phaseIITankClusterTree"]
        else:
            Event = file_1["Event"]
        
        Event = file_1["phaseIITankClusterTree"]
        EN = Event["eventNumber"].array()
        ETT = Event["eventTimeTank"].array()
        CT = Event["clusterTime"].array()
        CPE = Event["clusterPE"].array()
        CCB = Event["clusterChargeBalance"].array()
        CH = Event["clusterHits"].array()

        if which_Tree == 0:
            CN = Event["clusterNumber"].array()
            hT = Event["hitT"].array()
            hPE = Event["hitPE"].array()
            hID = Event["hitDetID"].array()
        else:
            CN = Event["numberOfClusters"].array()
            hT = Event["Cluster_HitT"].array()
            hPE = Event["Cluster_HitPE"].array()
            hID = Event["Cluster_HitDetID"].array()
            

        for i in trange(len(EN)):       # loop through aquisitions

            # 1. Check if the small AmBe PMT rejected the event

            # 2. Remove any events with a prompt (2us) clusters from the analysis
            #    This removes acquisitions triggered by through-going cosmic muons, whose Cherenkov light would also produce a PMT hit cluster in the tank

            # 3. Remove any acquisition containing a tank cluster with a cluster PE > 150 (could be thru-going) 

            # 4. Look for neutrons using normal selection criteria

            if ETT[i] in good_events:   # timestamp is found in the accepted aquisitions

                total_events += 1

                # loop through clusters array
                for k in range(len(CT[i])):

                    is_cosmic = cosmic(CT[i][k], CPE[i][k])

                    # if there isn't a cosmic, look for neutrons
                    if(is_cosmic==True):
                        cosmic_events += 1
                        break   # move onto the next event

                    is_neutron = AmBe(CPE[i][k], CCB[i][k], CT[i][k], CN[i])

                    if(is_neutron==True):
                        if CPE[i][k] != float('-inf'):
                            # AmBe neutron properties
                            cluster_time.append(CT[i][k])   
                            cluster_charge.append(CPE[i][k])
                            cluster_QB.append(CCB[i][k])
                            cluster_hits.append(CH[i][k])
                            hit_times.append(hT[i][k])
                            hit_charges.append(hPE[i][k])
                            hit_ids.append(hID[i][k])
                            source_position[0].append(x_pos)
                            source_position[1].append(y_pos)
                            source_position[2].append(z_pos)
                            neutron_cand_count += 1

    print('\nThere were a total of', total_events, 'AmBe events after the initial aquisition cuts, with', \
          cosmic_events, '(' , round(100*cosmic_events/total_events,2), '%) being triggered by cosmics')
    print('This leaves', total_events-cosmic_events, 'AmBe-triggered events\n')
    print('\nAfter selection cuts, we have: ', neutron_cand_count, ' AmBe neutron candidates\n')
    
    c1 += 1     # update loop

    
print('----------------------------------------------------------------\n')
print('We have: ', len(cluster_time), ' total AmBe neutron candidates\n')

print('\nWriting to root tree...')
os.system('rm ' + output_filename)
root_file = uproot.create(output_filename)

tree_data = {
    "cluster_time": cluster_time,
    "cluster_charge": cluster_charge,
    "cluster_Qb": cluster_QB,
    "cluster_hits": cluster_hits,
    "hitT": hit_times,
    "hitPE": hit_charges,
    "hitID": hit_ids,
    "X_pos": source_position[0],
    "Y_pos": source_position[1],
    "Z_pos": source_position[2]
}

root_file["Neutrons"] = tree_data
root_file.close()

print('\ndone\n')
