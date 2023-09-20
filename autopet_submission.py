import os
import torch
import model
import SimpleITK
import numpy as np


def nii2numpy(itk_obj):
    return SimpleITK.GetArrayFromImage(itk_obj)
class Model_infer(): 
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # .mha dir
        self.input_path = "/input/" 
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"  
        
        self.nii_path = "/opt/algorithm/"  # tmp path to store nifity files
        self.ckpt_path = "/opt/algorithm/epoch=777-step=64573.ckpt"

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        pass


    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)


    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)
        
        
    def convert_nii_to_npz(self, suv_in_path, ct_in_path):
        suv_arr = nii2numpy(SimpleITK.ReadImage(suv_in_path))
        ct_arr = nii2numpy(SimpleITK.ReadImage(ct_in_path))
        npz_path = suv_in_path.replace("SUV.nii.gz", "petct_arr")
        np.savez_compressed(npz_path, suv=suv_arr, ct=ct_arr)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

        
    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
      
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path,  'SUV.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path,  'CTres.nii.gz'))
        
        self.convert_nii_to_npz(os.path.join(self.nii_path, 'SUV.nii.gz'),
                                os.path.join(self.nii_path, 'CTres.nii.gz'))
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        self.convert_nii_to_mha(
            os.path.join(self.output_path, "PRED.nii.gz"), 
            os.path.join(self.output_path, uuid + ".mha"))
        
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))
    
    def predict(self, inputs):
        """
        Your algorithm goes here
        """        
        pass
        #return outputs

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()
        
        print('Start prediction')
        model.run_inference(self.ckpt_path, self.nii_path, self.output_path)
        
        print('Start output writing')
        self.write_outputs(uuid)


if __name__ == "__main__":
    Model_infer().process()