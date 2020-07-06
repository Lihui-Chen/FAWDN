# Description of Test Options

Let us take [`test_FAWDN_x4.json`](./test_FAWDN_x4.json) as an example. 

**Note**: Before you run `python test.py -opt options/test/*.json`, please carefully check options: `"scale"`, `"degradation"`,  `"self_ensemble"`, `"dataroot_HR"`, `"dataroot_LR"`, `"networks"` and `"pretrained_path"`.

```c++
{
    "mode": "sr", // solver type (only "sr" is provided)
    "use_cl": true, // whether use multiple losses (required by our SRFBN)
    "gpu_ids": [0], // GPU ID to use

    "scale": 4, // super resolution scale (*Please carefully check it*)
    "degradation": "BI", // degradation model for SR: "BI" | "BD" | "DN" (*Please carefully check it*)
    "is_train": false, // whether train the model
    "use_chop": true, // whether enable memory-efficient test
    "rgb_range": 255, // maximum value of images
    "self_ensemble": false, // whether use self-ensemble strategy
    
    // test dataset specifications (you can place more than one test dataset here) (*Please carefully check dateset mode/root*)
    "datasets": { 
        "test_set1": {
            "mode": "LRHR", // dataset mode: "LRHR" | "LR"
            "dataroot_HR": "./results/HR/MRI13/x4", // HR dataset root (required by "LRHR" dataset mode) 
            "dataroot_LR": "./results/LR/LRBI/MRI13/x4", // LR dataset root (required by "LRHR"/"LR" dataset mode) 
            "data_type": "img" // data type: "img" (image files) | "npy" (binary files), "npy" is recommended during training
        },
//        "test_set2": {
//             "mode": "LRHR",
//             "dataroot_HR": "./results/HR/Set14/x4",
//             "dataroot_LR": "./results/LR/LRBI/Set14/x4",
//             "data_type": "img"
//         },
        "test_set3": {
             "mode": "LR",
             "dataroot_LR": "./results/LR/MyImage",
             "data_type": "img"
         }
    },
    
    // networks specifications
    "networks": { 
        "which_model": "FAWDN", // network name
        "num_features": 16,  // number of growth rate
        "in_channels": 3, // number of input channels. If input image is gray image, it will be copied three times and concatenated along the channel direction.
        "out_channels": 3, // number of output channels
        "num_steps": 2, // number of time steps (T)
        "nDenselayer": 8, // number of dense layer in AWDB
        "nBlock": 8 // number of AWDB in the hinden unit
    },
    
    "solver": {
        "pretrained_path": "./models/models/best_FAWDN+_x4.pth" // pre-trained model directory (for test)
    }
}
```
