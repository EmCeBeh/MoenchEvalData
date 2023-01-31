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

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class DaskSetup:
    def da_imsave(fnames, arr, compute=False):
        """Write arr to a stack of images assuming
        the last two dimensions of arr as image dimensions.

        Parameters
        ----------
        fnames: string
            A formatting string like 'myfile{:02d}.png'
            Should support arr.ndims-2 indices to be formatted
        arr: dask.array
            Array of at least 2 dimensions to be written to disk as images
        compute: Boolean (optional)
            whether to write to disk immediately or return a dask.array of the to be written indices

        """
        indices = [da.arange(n, chunks=c) for n,c in zip(arr.shape[:-2], arr.chunksize[:-2])]
        index_array = da.stack(da.meshgrid(*indices,indexing='ij'), axis=-1).rechunk({-1:-1})

        @da.as_gufunc(signature=f"(i,j),({arr.ndim-2})->({arr.ndim-2})", output_dtypes=int, vectorize=True)
        def saveimg(image, index):
            im = Image.fromarray(image.squeeze().astype(np.uint32))
            im.save(fnames.format(*index))
            return index

        res = saveimg(arr,index_array)
        if compute == True:
            res.compute()
        else:
            return res     
        
        @da.as_gufunc(signature=f"(i,j),(),()->(i,j)", output_dtypes=int, vectorize=True)
        def max_of_clusters_var_cl_th_counts(img, cl, th):
            mask = np.ones([cl,cl])
            clusters_scipy = convolve2d(img, mask, mode='same', fillvalue=0)
            maxima = peak_local_max(clusters_scipy, min_distance=cl, threshold_abs=th)
            max_value_mask = np.zeros([400,400])
            max_value_mask[maxima[:,0], maxima[:,1]] = 1
            max_clusts = clusters_scipy * max_value_mask
            counts = max_value_mask.sum(axis=(0,1))
            return max_clusts
        
        @da.as_gufunc(signature=f"(i,j),(),(),()->(i,j)", output_dtypes=int, vectorize=True, allow_rechunk=True)
        def single_photon_mask_var_cl_th_dis(img, cl, th, dis): # dis must be at least 1 
            mask = np.ones([cl,cl])
            clusters_scipy = convolve2d(img, mask, mode='same', fillvalue=0)
            maxima = peak_local_max(clusters_scipy, min_distance=dis, threshold_abs=th)
            mask = np.zeros([400,400])
            mask[tuple(maxima.T)] = True
            return mask
        
        @da.as_gufunc(signature=f"(i,j),()->(i,j)", output_dtypes=int, vectorize=True, allow_rechunk=True)
        def single_photon_mask_ver(img, th):
            mask_ver = np.ones([2,1])
            clusters_scipy_ver = convolve2d(img, mask_ver, mode='same', fillvalue=0)
            maxima_ver = peak_local_max(clusters_scipy_ver, min_distance=1, threshold_abs=th)
            mask = np.zeros([400,400])
            mask[tuple(maxima_ver.T)] = True
            return mask
        
        @da.as_gufunc(signature=f"(i,j),()->(i,j)", output_dtypes=int, vectorize=True, allow_rechunk=True)
        def single_photon_mask_hor(img, th):
            mask_hor = np.ones([1,2])
            clusters_scipy_hor = convolve2d(img, mask_hor, mode='same', fillvalue=0)
            maxima_hor = peak_local_max(clusters_scipy_hor, min_distance=1, threshold_abs=th)
            mask = np.zeros([400,400])
            mask[tuple(maxima_hor.T)] = True
            return mask
        
        
        @da.as_gufunc(signature=f"(i,j),()->(i,j)", output_dtypes=int, vectorize=True, allow_rechunk=True)
        def single_photon_mask_ver_hor(img, th):
            img_pos = (img > 0) * img
            mask_ver = np.ones([2,1])
            mask_hor = np.ones([1,2])
            clusters_scipy_ver = convolve2d(img_pos, mask_ver, mode='same', fillvalue=0)
            clusters_scipy_hor = convolve2d(img_pos, mask_hor, mode='same', fillvalue=0)
            maxima_ver = peak_local_max(clusters_scipy_ver, min_distance=1, threshold_abs=th)
            maxima_hor = peak_local_max(clusters_scipy_hor, min_distance=1, threshold_abs=th)
            mask_ver = np.zeros([400,400])
            mask_ver[tuple(maxima_ver.T)] = True
            mask_hor = np.zeros([400,400])
            mask_ver[tuple(maxima_hor.T)] = True
            mask = np.logical_or(mask_ver, mask_hor)
            return mask


class DataEvaluation:
    def __init__(self,
                 day, month, year,
                 dask_scheduler_address,
                 mask=None,
                 px_size=25e-6,
                 dist=0.620,
                 wavelength = 1.8e-9,
                 npt = 40,
                 qi  = 7,
                 qf  = 22,
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
        
        self.qi = qi
        self.qf = qf
        
        
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
        path_unpumped     = path_base + r'*[0,2,4,6,8].tiff' # example path
        path_pumped   = path_base + r'*[1,3,5,7,9].tiff' # example path

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
                
                mask_unpumped = (unpumped > t[0]) & (unpumped < t[1])
                mask_pumped = (pumped > t[0]) & (pumped < t[1])

                #apply masks
                unpumped = unpumped * mask_unpumped
                pumped   = pumped * mask_pumped

            #mean and compute
            unpumped = unpumped.mean(axis=0).compute()
            pumped   = pumped.mean(axis=0).compute()

        if mode == 'counting':
            
            raise ValueError('Not implemented yet!')
            
            if t is not None:
                counting_unpumped = single_photon_mask_ver_hor(unpumped, t[0])
                counting_pumped = single_photon_mask_ver_hor(pumped, t[0])
                unpumped = counting_unpumped.mean(axis=0).compute()
                pumped   = counting_pumped.mean(axis=0).compute()


        return unpumped, pumped

    def check_image(self, num):
        
        # todo: change axis to mm via self.px_size or to q...
        
        unpumped, pumped = self.get_images(num)
        plt.figure()
        for i, image in enumerate([unpumped, pumped]):

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
    
    
    def ai_single(self, num):
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
            
            I_unpumped      : 1-d array; un-normalised intensities for unpumped images 
            I_pumped        : 1-d array; un-normalised intensities for pumped images 
            
            norm_unpumped   : 1-d array; intensities normalised to direct beam (inner mask part) for unpumped images 
            norm_pumped     : 1-d array; intensities normalised to direct beam (inner mask part) for pumped images 
        
        """
        unpumped, pumped = self.get_images(num)
        q, I_unpumped = self.ai.integrate1d(unpumped, npt = self.npt, mask=self.mask)
        q, I_pumped = self.ai.integrate1d(pumped, npt = self.npt, mask=self.mask)

        # searching for q indexes where q is less than q1 [um^-1] and greater than q2 [um^-1]
        # thus q1 and q2 are divided by 1000, pyfai output is in nm^-1...
        indexes_q = np.argwhere(np.logical_and(q >= self.qi/1000, q <= self.qf/1000))

        q = q[indexes_q]
        I_unpumped = I_unpumped[indexes_q]
        I_pumped = I_pumped[indexes_q]
        

        
        I_unpumped_norm = (unpumped * self.mask).sum()      # change to inner part only!!!!!
        I_pumped_norm = (pumped * self.mask).sum()      # change to inner part only!!!!!
        
        return q, I_unpumped, I_pumped, I_unpumped_norm, I_pumped_norm
    

        
    def check_q_integration(self, num, norm = False):
        
        q, I_unpumped, I_pumped, I_unpumped_norm, I_pumped_norm = self.ai_single(num)
        
        
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
    

    def integrated_series(self, nums):
        """
        Takes numbers of scans and returns the integrated q-vectors for unpumped and pumped AIs
        
        returns:  1-d array
        """
    
        list_I_unpumped = []
        list_I_pumped = []
        list_I_unpumped_norm = []
        list_I_pumped_norm = []

        for num in tqdm(nums):
            q, I_unpumped, I_pumped, I_unpumped_norm, I_pumped_norm = self.ai_single(num)
            list_I_unpumped.append(np.copy(I_unpumped))
            list_I_pumped.append(np.copy(I_pumped))
            list_I_unpumped_norm.append(np.copy(I_unpumped_norm))
            list_I_pumped_norm.append(np.copy(I_pumped_norm))

        array_I_unpumped = np.array(list_I_unpumped)
        array_I_pumped = np.array(list_I_pumped)
        array_I_unpumped_norm = np.array(list_I_unpumped_norm)
        array_I_pumped_norm = np.array(list_I_pumped_norm)


        I_unpumped_mean = array_I_unpumped.mean(axis=1)
        I_pumped_mean = array_I_pumped.mean(axis=1)

        return I_unpumped_mean, I_pumped_mean
