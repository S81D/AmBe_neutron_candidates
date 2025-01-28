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

# ----------------------------------------------------- #

# AmBe neutrons
def AmBe(CPE, CCB, CT):
    if(CPE<=0 or CPE>70):      # 0 < cluster PE < 100
        return False
    if(CCB>=0.4 or CCB<=0):   # Cluster Charge Balance < 0.4
        return False
    if(CT<2000):              # cluster time not in prompt window
        return False
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
    
    # Port 5 data
    if run == '4506':
        xp = 0; yp = 100; zp = 0
    elif run == '4505':
        xp = 0; yp = 50; zp = 0
    elif run == '4499':
        xp = 0; yp = 0; zp = 0
    elif run == '4507':
        xp = 0; yp = -50; zp = 0
    elif run == '4508':
        xp = 0; yp = -100; zp = 0
        
    # Port 1 data
    elif run == '4593':
        xp = 0; yp = 100; zp = -75
    elif run == '4590':
        xp = 0; yp = 50; zp = -75
    elif run == '4589':
        xp = 0; yp = 0; zp = -75
    elif run == '4596':
        xp = 0; yp = -50; zp = -75
    elif run == '4598':
        xp = 0; yp = -100; zp = -75
        
    # Port 4 data
    elif run == '4656':
        xp = -75; yp = 100; zp = 0
    elif run == '4658':
        xp = -75; yp = 50; zp = 0
    elif run == '4660':
        xp = -75; yp = 0; zp = 0
    elif run == '4662' or run == '4664' or run == '4665' or run == '4666' or run == '4667' or run == '4670':
        xp = -75; yp = -50; zp = 0
    elif run == '4678' or run == '4683' or run == '4687':
        xp = -75; yp = -100; zp = 0
        
    else:
        print('\n##### RUN NUMBER ' + run + 'DOESNT HAVE A SOURCE LOCATION!!! ERROR #####\n')
        exit()
        
    return xp, yp, zp

    
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

pulse_thresh = 120     # cutoff for "good" pulses, in units of IC

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

                    # check if the pulse is sufficiently large
                    if IC_adjusted > pulse_thresh:
                        
                        post_pulse_mask = hist_edges[:-1] > pulse_end
                        post_pulse_values = hist_values[post_pulse_mask]
                        another_pulse = np.any(post_pulse_values > (5 * sigma + baseline))
                        
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


    # Second step, load data
    print('\nLoading AmBe event data...')

    cosmic_events = 0; total_events = 0; neutron_cand_count = 0
    with uproot.open(file_names[c1]) as file_1:     # c1 iterated through each loop
        
        Event = file_1["phaseIITankClusterTree"]
        EN = Event["eventNumber"].array()
        ETT = Event["eventTimeTank"].array()
        CT = Event["clusterTime"].array()
        CPE = Event["clusterPE"].array()
        CCB = Event["clusterChargeBalance"].array()
        CH = Event["clusterHits"].array()
        hT = Event["hitT"].array()
        hPE = Event["hitPE"].array()
        hID = Event["hitDetID"].array()

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

                    is_neutron = AmBe(CPE[i][k], CCB[i][k], CT[i][k])

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
