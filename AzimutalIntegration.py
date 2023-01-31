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

class DataEvaluation:
    def __init__(self, day, month, year, dask_scheduler_address, mask=None, px_size=25e-6, dist=0.620, wavelength = 1.8e-9, center=(185,210)):
        """
        
        """
        detector = Detector(px_size, px_size)  # binned!
        self.ai = AzimuthalIntegrator(dist=dist, #Distanz Detektor Probe
                                     poni2=center[0] * px_size, #center of the beam, actually... But I have not yet found out which values are transferred there and how.
                                     poni1=center[1] * px_size, #center of ray2, just try around until your scattering ring is straight and your q-space starts at 0
                                     detector=detector, #detector you have previously defined
                                     wavelength=wavelength) # wavelength of x-ray beam
        self.day = day
        self.month = month
        self.year = year
        
        
        if mask is None:
            mask = np.ones([400,400])
            mask[40:360,40:360] = 0
            mask[170:249, 152:269] = 1
            self.mask = mask
        print("")
        self.client = Client(dask_scheduler_address)

    def load_dark(self, dark_num, d=None, m=None, y = None, path = "/mnt/temp_nvme_ssd/", verbose = False):
        if y is None:
            y  = self.year
        if m is None:
            m = self.month
        if d is None:
            d = self.day
        date = '%04d%02d%02d'%(y,m,d)

        path_dark     = r'%s/%s/%s/*'%(path,date,dark_num)
        dark_images = dask_image.imread.imread(path_dark, arraytype="numpy")
        avg_dark = np.average(dark_images, axis=0).compute()

        if verbose:
            dark_images_1000 = (dark_images[:1000] - avg_dark).compute()
            print(f'average {np.average(dark_images_1000)} std {np.std(dark_images_1000)}')
        self.darks = [avg_dark, avg_dark]

    def get_images(self, num, invert = False, mode = 'analog', t = (100, 200), d = None, m = None, y = None, path = "/mnt/temp_nvme_ssd/", verbose = False):
        if y is None:
            y  = self.year
        if m is None:
            m = self.month
        if d is None:
            d = self.day
        date = '%04d%02d%02d'%(y,m,d) 

        base = r'%s/%s/%s/'%(path,date,num)
        path_all      = base + r'*'
        path_pumped = base + r'*[0,2,4,6,8].tiff' # example path
        path_unpumped   = base + r'*[1,3,5,7,9].tiff' # example path

        if verbose:
            print(path)
            print(path_all)
            print(path_unpumped)
            print(path_pumped)

        # imgs_all = dask_image.imread.imread(path_all,arraytype="numpy")
        imgs_unpumped = dask_image.imread.imread(path_unpumped,arraytype="numpy")
        imgs_pumped = dask_image.imread.imread(path_pumped,arraytype="numpy")

        if verbose:
            # print(imgs_all.shape)
            print(imgs_unpumped.shape)
            print(imgs_pumped.shape)

        if invert:
            imgs_unpumped, imgs_pumped = imgs_pumped, imgs_unpumped

        if dark is not None:
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

        if mode == 'develop':
            if t is not None:
                counting_pumped = single_photon_mask_ver_hor(pumped, t[0])
                counting_unpumped = single_photon_mask_ver_hor(unpumped, t[0])
                pumped   = counting_pumped.mean(axis=0).compute()
                unpumped = counting_unpumped.mean(axis=0).compute()


        return unpumped, pumped
    
    def ai_single(self, number, d = None, m = None, y = None, t = (100, 200), q0 = 10, q1 = 20):
        if y is None:
            y  = self.year
        if m is None:
            m = self.month
        if d is None:
            d = self.day
        """ Azimuthal Integrator for single SAXS images
        Takes image number (number), the azimuthal integrator (ai), mask (None as default) and date (d) as input.
        Gets pumped and unpumped images with get_images funtion
        Calculates the azimuthal 1d intensities of the given image
        Returns q-vector, the 1d Intensities for pumped and unpumped images, normalized to direct beam Intensities
        """
        unpumped, pumped = self.get_images(number, t=t, d=d)
        q, I_pump = self.ai.integrate1d(pumped, 40, mask=self.mask)
        q, I_unpump = self.ai.integrate1d(unpumped, 40, mask=self.mask)

        # searching for q indexes where q is less than q1 [um^-1] and greater than q2 [um^-1]
        # thus q1 and q2 are divided by 1000, pyfai output is in nm^-1...
        indexes_q = np.argwhere(np.logical_and(q >= q0/1000, q <= q1/1000))

        q = q[indexes_q]
        I_pump = I_pump[indexes_q]
        I_unpump = I_unpump[indexes_q]

        norm_pumped = (pumped * mask).sum()
        norm_unpumped = (unpumped * mask).sum()


        return q, I_pump, I_unpump, norm_pumped, norm_unpumped
    
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
