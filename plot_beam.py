#!/usr/bin/env python
# ian.heywood@physics.ox.ac.uk


import numpy
import os
import pylab
import sys

from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from optparse import OptionParser
from scipy import ndimage,stats


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


fontpath = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
if os.path.isfile(fontpath):
    import matplotlib
    import matplotlib.font_manager as font_manager
    #matplotlib.font_manager._rebuild()
    prop = font_manager.FontProperties(fname=fontpath)
    matplotlib.rcParams['font.family'] = prop.get_name()


def set_fontsize(fig,fontsize):
    def match(artist):
        return artist.__module__ == "matplotlib.text"

    for textobj in fig.findobj(match=match):
        textobj.set_fontsize(fontsize)


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def get_image(fitsfile):
        input_hdu = fits.open(fitsfile)[0]
        if len(input_hdu.data.shape) == 2:
                image = numpy.array(input_hdu.data[:,:])
        elif len(input_hdu.data.shape) == 3:
                image = numpy.array(input_hdu.data[0,:,:])
        elif len(input_hdu.data.shape) == 4:
                image = numpy.array(input_hdu.data[0,0,:,:])
        else:
                image = numpy.array(input_hdu.data[0,0,0,:,:])
        return image


def deg2rad(xx):
    return numpy.pi*xx/180.0


def get_beam(fitsfile):
    input_hdu = fits.open(fitsfile)[0]
    hdr = input_hdu.header
    bmaj = hdr.get('BMAJ')
    bmin = hdr.get('BMIN')
    bpa = hdr.get('BPA')
    pixscale = hdr.get('CDELT2')
    return bmaj,bmin,bpa,pixscale


def crop_image_centre(img,dx,dy):
    y,x = img.shape
    x0 = x//2 - dx//2
    y0 = y//2 - dy//2    
    return img[y0:y0+dy, x0:x0+dx]

def crop_image(img,x0,y0,dx,dy):
    if x0 < 0 and y0 < 0:
        y,x = img.shape
        x0 = x//2 - dx//2
        y0 = y//2 - dy//2    
    else:        
        x0 = x0 - dx//2
        y0 = y0 - dy//2    
    return img[y0:y0+dy, x0:x0+dx]

def evaluate_gaussian(x_values,mean,fwhm,pixscale):
        sigma = fwhm/(2.3548*pixscale)
        y_values = stats.norm(mean,sigma).pdf(x_values)
        y_values = y_values / numpy.max(y_values)
        return y_values


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def main():


    # SETUP OPTIONS
    parser = OptionParser(usage = '%prog [options] msname')
    parser.add_option('--x0', dest = 'x0', default = -1, help = 'X pixel of PSF peak (default = central pixel)')
    parser.add_option('--y0', dest = 'y0', default = -1, help = 'Y pixel of PSF peak (default = central pixel)')
    parser.add_option('--cropsize', dest = 'cropsize', default = 51, help = 'Size of region to extract in pixels (default = 51)')
    
    (options,args) = parser.parse_args()
    x0 = int(options.x0)
    y0 = int(options.y0)
    cropsize = options.cropsize

    if len(args) != 1:
        print('Please specify a PSF FITS image')
        sys.exit()
    else:
        psf_fits = args[0].rstrip('/')

    pngname = 'psfplot_'+psf_fits.split('/')[-1]+'.png'


    # BEAM IMAGE
    bmaj,bmin,bpa,pixscale = get_beam(psf_fits)
    psf_image = get_image(psf_fits)
    psf_image = crop_image(psf_image,x0,y0,cropsize,cropsize)
    psf_image_rot = ndimage.rotate(psf_image,bpa,reshape=False)
    psf_image_rot = psf_image_rot / numpy.max(psf_image_rot) # Normalised in case of killMS usage


    # FITTED BEAM IMAGE
    xstd = bmin/(2.3548*pixscale)
    ystd = bmaj/(2.3548*pixscale)
    theta = deg2rad(bpa)
    restoring = Gaussian2DKernel(x_stddev=xstd,y_stddev=ystd,theta=theta,x_size=cropsize,y_size=cropsize,mode='center')
    restoring_image = restoring.array
    restoring_image = restoring_image / numpy.max(restoring_image)
    restoring_image_rot = ndimage.rotate(restoring_image,bpa,reshape=False)


    # BEAM AND FITTED BEAM SLICES
    slice_indices = numpy.arange(0,cropsize)
    slice_maj = psf_image_rot[:,cropsize//2]
    fitted_slice_maj = evaluate_gaussian(slice_indices,cropsize//2,bmaj,pixscale)
#    fitted_slice_maj = kernel_rot[:,cropsize//2]
    slice_min = psf_image_rot[cropsize//2,:]
    fitted_slice_min = evaluate_gaussian(slice_indices,cropsize//2,bmin,pixscale)


    # MAKE THE PLOT
    fig = pylab.figure(figsize=(20,20))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(slice_indices,slice_min,'-',color='skyblue',label='Minor axis slice')
    ax1.plot(slice_indices,fitted_slice_min,'-',color='hotpink',label='Fitted')
    ax1.plot(slice_indices,slice_min-fitted_slice_min,'-',color='orange',label='Residual')
    ax1.plot([0,cropsize],[0.0,0.0],'-',linestyle='dashed',color='black',alpha=0.6,zorder=100)
    ax1.legend()
    ax1.set_title('Minor axis')

    ax2.imshow(psf_image_rot-restoring_image_rot,vmin=-0.05,vmax=0.2,cmap='Greys',interpolation='bicubic')
    ax2.set_title('Residual')

    ax3.imshow(psf_image_rot,vmin=-0.05,vmax=0.2,cmap='Greys',interpolation='bicubic')
    ax3.contour(psf_image_rot,levels=[0.5],colors='skyblue',linewidths=[3])
    ax3.plot([cropsize//2,cropsize//2],[0,cropsize-1],'-',color='skyblue',lw=3)
    ax3.plot([0,cropsize-1],[cropsize//2,cropsize//2],'-',color='skyblue',lw=3)
    ax3.set_title('PSF image with 0.5 contour')

    ax4.plot(slice_maj,slice_indices,'-',color='skyblue',label='Major axis slice')
    ax4.plot(fitted_slice_maj,slice_indices,'-',color='hotpink',label='Fitted')
    ax4.plot(slice_maj-fitted_slice_maj,slice_indices,'-',color='orange',label='Residual')
    ax4.plot([0.0,0.0],[0,cropsize],'-',linestyle='dashed',color='black',alpha=0.6,zorder=100)
    ax4.legend()
    ax4.set_ylim(ax4.get_ylim()[::-1])
    ax4.set_title('Major axis')

    fig.suptitle(psf_fits)
    set_fontsize(fig,24)
    fig.savefig(pngname)


if __name__ == '__main__':

    main()