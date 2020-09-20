import os
import json
import sys
import fastai
from fastai.vision import *
import warnings
warnings.filterwarnings("ignore")

path = Path(os.getcwd())

learn = load_learner(path)

checkpath = Path(os.path.join(os.getcwd(), sys.argv[1]))

# check path existing
if not os.path.exists(checkpath):
    # it means that it is the absolute path
    checkpath = Path(sys.argv[1])
    # check abs path existing
    if not os.path.exists(checkpath):
        print("Path doesn't exist")
        
namelist = os.listdir(checkpath)

# temporary prediction list
predictlist = []

# make predictions
for i in os.listdir(checkpath):
    img = open_image(checkpath/i)
    pred_class,pred_idx,outputs = learn.predict(img)
    predictlist.append(str(pred_class))
    
# make a final dict {filename: prediction, ...}
result = {k:v for k,v in zip(namelist, predictlist)}

# write to .json file
json_object = json.dumps(result, indent = 4) 
with open("process_results.json", "w") as outfile: 
    outfile.write(json_object)
    
print('Done. json file is ready')