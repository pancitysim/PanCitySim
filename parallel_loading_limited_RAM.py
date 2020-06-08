import time
import threading
import multiprocessing


print (multiprocessing.cpu_count(), " Thread count available ")
graphs_dump = {}
# load first 10
for i in range(10):
    graphs_dump[i] = "Available"

# Inspired by from https://amalgjose.com/2018/07/18/run-a-background-function-in-python/

def background(f):
    '''
    a threading decorator
    use @background above the function you want to run in the background
    '''
    def backgrnd_func(*a, **kw):
        threading.Thread(target=f, args=a, kwargs=kw).start()
    return backgrnd_func


def mp_worker(key):
    # load graph at time "key"
    # let us say this loading takes 5 seconds
    time.sleep(5)
    read_data = "Available"  # read into a separate variable; then load into graophs to avoid locking time; 
    # not sure if this is required; but just a safety measure
    return (key,read_data)

def mp_handler(data):
    p = multiprocessing.Pool(2)
    res = p.map(mp_worker, data)
    
    # populating the global dict here
    global graphs_dump
    for k_v in res:
        graphs_dump[k_v[0]] = k_v[1]
    p.close()
    p.join()
    
@background
def updateDict(t):
    #This will print the count for every second
    # graphs_dump is a global variable consisting of the graphs at different times of day
    gd_keys = list(graphs_dump.keys())
    
    # remove the used graphs (before time t)
    for key in gd_keys :
        if key < t:
            del graphs_dump[key]
    
    # get new ones (upto t+10)
    data = []
    for key in range(t,t+10):
        if key not in graphs_dump:
            data.append(key)
    
    # removing already processed t to avoid creating duplicate processes
    dd = []
    global already_processed_t

    for key in data:
        if key not in already_processed_t:
            dd.append(key)
    data = dd
    
    already_processed_t = already_processed_t + data
    if len(data) > 0:
        mp_handler(data)


globalStartTime = time.time()

already_processed_t = []
for day_num in range(100):
    already_processed_t = []
    for t in range(1440):
        updateDict(t)
        
        startTime = time.time()
        while(t not in graphs_dump):
            time.sleep(0.5)
            print ("Waiting for ", time.time() - startTime, " seconds @ time slot ", t)
        
        ## process graph at current t
        ## let us say this processing takes 1 second
        time.sleep(1)
        print ("Time slots currently in RAM ",list(graphs_dump.keys()))
        print ("Graph @ time slot ", t , " processed in ",time.time() - globalStartTime," seconds \n")
    
