import pyFAI
pyFAI.disable_opencl=True # get rid of annoying warning ;)
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector
import numpy as np
import dask_image.imread
import dask_image.ndfilters
import dask
import dask.array as da
from dask.distributed import Client

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class DataEvaluation:
    def __init__(self,
                 day, month, year,
                 dask_scheduler_address,
                 mask=None,
                 px_size=25e-6,
                 dist=0.620,
                 wavelength = 1.8e-9,
                 npt = 40,
                 center=(185,210),
                 storage_path = "/mnt/temp_nvme_ssd",
                 invert = False):
        """
                        invert : Boolean;   if False (dafault), even numbers are unpumped images, odd numbers are pumped images
                                    if True, order is inverted
        """
        self.center = center
        self.px_size = px_size
        detector = Detector(px_size, px_size)  # binned!
        self.ai = AzimuthalIntegrator(dist=dist, #Distanz Detektor Probe
                                     poni2=self.center[0] * px_size, #center of the beam, actually... But I have not yet found out which values are transferred there and how.
                                     poni1=self.center[1] * px_size, #center of ray2, just try around until your scattering ring is straight and your q-space starts at 0
                                     detector=detector, #detector you have previously defined
                                     wavelength=wavelength) # wavelength of x-ray beam
        
        self.npt = npt
        
        
        self.day = day
        self.month = month
        self.year = year
        self.invert = invert
        
        self.storage_path = storage_path
        self.date = '%04d%02d%02d'%(self.year,self.month,self.day)
        self.path     = r'%s/%s'%(self.storage_path,self.date)
        
        
        self.darks = None        
        self.log_norm = LogNorm(0,150)
        
        
        
        
        
        if mask is None:
            mask = np.ones([400,400])
            mask[40:360,40:360] = 0
            mask[170:249, 152:269] = 1
            self.mask = mask
        print("")
        self.client = Client(dask_scheduler_address)

    def update_dark(self, num, verbose = False):
        """
            function for loading dark images, which should not be split into pumped and unpumped (e.g. pedestals)
        """
        path_dark     = r'%s/%s/*'%(self.path, num)
        dark_images   = dask_image.imread.imread(path_dark, arraytype="numpy")
        avg_dark      = np.average(dark_images, axis=0).compute()

        if verbose:
            dark_images_1000 = (dark_images[:1000] - avg_dark).compute()
            print(f'average {np.average(dark_images_1000)} std {np.std(dark_images_1000)}')
        self.darks = [avg_dark, avg_dark]

    def get_images(self, num, mode = 'analog', t = (100, 200), verbose = False):
        '''
            input:
                num    : Integer;   file number of the image

            
            returns:
            
            unpumped   :  np Array; Dimension equal to dimension of given images (e.g. for PSI Jungfrau MOENCH detector natively 400x400)
            pumped     :  np Array; Dimension equal to dimension of given images (e.g. for PSI Jungfrau MOENCH detector natively 400x400)
        '''

         

        path_base       = r'%s/%s/'%(self.path,num)
        path_pumped     = path_base + r'*[0,2,4,6,8].tiff' # example path
        path_unpumped   = path_base + r'*[1,3,5,7,9].tiff' # example path

        if verbose:
            print(self.path)
            print(path_base)
            print(path_pumped)
            print(path_unpumped)
            

        imgs_unpumped = dask_image.imread.imread(path_unpumped,arraytype="numpy")
        imgs_pumped = dask_image.imread.imread(path_pumped,arraytype="numpy")

        if verbose:
            print(imgs_unpumped.shape)
            print(imgs_pumped.shape)

        if self.invert:
            imgs_unpumped, imgs_pumped = imgs_pumped, imgs_unpumped

        if self.darks is not None:
            unpumped = imgs_unpumped - self.darks[0]
            pumped   = imgs_pumped - self.darks[1]

        else:
            unpumped = imgs_unpumped
            pumped   = imgs_pumped

        if mode == 'analog':

            if t is not None:

                # boolean masks
                mask_pumped = (pumped > t[0]) & (pumped < t[1])
                mask_unpumped = (unpumped > t[0]) & (unpumped < t[1])

                #apply masks
                pumped   = pumped * mask_pumped
                unpumped = unpumped * mask_unpumped

            #sum, and compute
            pumped   = pumped.mean(axis=0).compute()
            unpumped = unpumped.mean(axis=0).compute()

        if mode == 'counting':
            
            raise ValueError('Not implemented yet!')
            
            if t is not None:
                counting_pumped = single_photon_mask_ver_hor(pumped, t[0])
                counting_unpumped = single_photon_mask_ver_hor(unpumped, t[0])
                pumped   = counting_pumped.mean(axis=0).compute()
                unpumped = counting_unpumped.mean(axis=0).compute()


        return unpumped, pumped
    
    def ai_single(self, num, qi = 10, qf = 20):
        """ Azimuthal Integrator for single SAXS images
        
        Gets pumped and unpumped images with get_images funtion
        Calculates the azimuthal 1d intensities of the given image
        Returns q-vector, the 1d Intensities for pumped and unpumped images, normalized to direct beam Intensities
        
        norm values ARE CURRENTLY STILL INCORRECT, if your mask has an outer part aswell!!!!
        
        Input:
            num    : integer;   image number
            qi     : float;     lowest (initial) q vector
            qf     : float;     highest (final) q vector
        
        Returns
            q               : 1-d array; q vectors
            I_pumped        : 1-d array; un-normalised intensities for pumped images 
            I_unpumped      : 1-d array; un-normalised intensities for unpumped images 
            norm_pumped     : 1-d array; intensities normalised to direct beam (inner mask part) for pumped images 
            norm_unpumped   : 1-d array; intensities normalised to direct beam (inner mask part) for unpumped images 
        
        """
        unpumped, pumped = self.get_images(num)
        q, I_pump = self.ai.integrate1d(pumped, npt = self.npt, mask=self.mask)
        q, I_unpump = self.ai.integrate1d(unpumped, npt = self.npt, mask=self.mask)

        # searching for q indexes where q is less than q1 [um^-1] and greater than q2 [um^-1]
        # thus q1 and q2 are divided by 1000, pyfai output is in nm^-1...
        indexes_q = np.argwhere(np.logical_and(q >= qi/1000, q <= qf/1000))

        q = q[indexes_q]
        I_pumped = I_pump[indexes_q]
        I_unpumped = I_unpump[indexes_q]

        I_pumped_norm = (pumped * self.mask).sum()      # change to inner part only!!!!!
        I_unpumped_norm = (unpumped * self.mask).sum()      # change to inner part only!!!!!
        
        return q, I_pumped, I_unpumped, I_pumped_norm, I_unpumped_norm
    
    def check_image(self, num):
        
        # todo: change axis to mm via self.px_size or to q...
        
        unpumped, pumped = self.get_images(num)
        plt.figure()
        for i, image in enumerate([pumped, unpumped]):

            plt.subplot(1,2,i+1)
            plt.imshow(image, norm=self.log_norm, interpolation="none")
            plt.imshow(self.mask, alpha=0.5)
            plt.plot(self.center[0], self.center[1], ".", markersize=10, color = 'r', alpha = 0.3)
            plt.axvline(self.center[0], color = 'r', alpha = 0.3)
            plt.axhline(self.center[1], color = 'r', alpha = 0.3)
            plt.title('%s'%['Unpumped','Pumped'][i])
            plt.xlabel('x-coordinate / px')
            plt.ylabel('y-coordinate / px')
            plt.tight_layout()
        plt.show()
        
    def check_q_integration(self, num, norm = False):
        
        q, I_pumped, I_unpumped, I_pumped_norm, I_unpumped_norm = self.ai_single(num)
        
        
        if norm:
            I2plot_unpumped = I_unpumped_norm
            I2plot_pumped = I_pumped_norm
        else:
            I2plot_unpumped = I_unpumped
            I2plot_pumped   = I_pumped
        
        plt.figure()
        plt.plot(q, I2plot_unpumped, label = 'unpumped', color = 'b')
        plt.plot(q, I2plot_pumped,   label = 'pumped',   color = 'r')
        plt.xlabel('scattering vector / (1/nm)')
        plt.ylabel('Intensity')                   
        plt.legend()
        
    

    
    def integrated_series(self, numbers, d = None, m = None, y = None, t = (100, 200), q0 = 10, q1 = 20):
        """ Takes numbers of scans and returns 
        """
        if y is None:
            y  = self.year
        if m is None:
            m = self.month
        if d is None:
            d = self.day
    
        list_ai_pump = []
        list_ai_unpump = []
        list_norm_pump = []
        list_norm_unpump = []

        for num in tqdm(numbers):
            q, ip, iup, norm_p, norm_up = ai_single(num, mask, d=d, q0=q0, q1=q1)
            list_ai_pump.append(np.copy(ip))
            list_ai_unpump.append(np.copy(iup))
            list_norm_pump.append(np.copy(norm_p))
            list_norm_unpump.append(np.copy(norm_up))

        ai_pump = np.array(list_ai_pump)
        ai_unpump = np.array(list_ai_unpump)
        norm_pump = np.array(list_norm_pump)
        norm_unpump = np.array(list_norm_unpump)


        peak_pump = ai_pump.mean(axis=1)
        peak_unpump = ai_unpump.mean(axis=1)

        return peak_pump, peak_unpump
