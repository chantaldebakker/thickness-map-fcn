import SimpleITK as sitk
import sys
import os

def GetThicknessMap(in_im):
	#Returns a thickness map of the input image

	#compute distance transform, then use it to calculate thickness of fracture site
	dt = sitk.SignedMaurerDistanceMapImageFilter()
	dt.InsideIsPositiveOn()
	#dt.UseImageSpacingOn()
	dt.SquaredDistanceOff()
	#cropped_fx = sitk.BinaryThreshold(cropped_fx, lowerThreshold=126, upperThreshold=150, insideValue=1, outsideValue=0)
	distancemap = dt.Execute(in_im)*cast.Execute(in_im)

	sitk.WriteImage(distancemap, 'distancemap.mha')

	#Calculate maximum distance to know how far to dilate: 
	stats2 = sitk.LabelStatisticsImageFilter()
	cca = sitk.ConnectedComponentImageFilter()
	im_label = cca.Execute(in_im)
	stats2.Execute(distancemap, im_label)
	max_distance = stats2.GetMaximum(1)


	#iterate through each distance value, putting a sphere with value equal to the distance
	#at the center of the sphere, and then selecting the largest distance value (= largest sphere
	#that the voxel fits within) for each voxel, then multiply by 2 and add 1 to convert to thickness

	bin_dil = sitk.BinaryDilateImageFilter()
	bin_dil.SetKernelType(sitk.sitkBall)
	dil1 = sitk.GrayscaleDilateImageFilter()
	dil1.SetKernelType(sitk.sitkBall)
	xor = sitk.XorImageFilter()

	for i in range(1, int(round(max_distance))+2): 

		im = sitk.BinaryThreshold(distancemap, lowerThreshold=i-.5, upperThreshold=i+.5, insideValue=1, outsideValue=0)
		im = cast.Execute(im)*distancemap

		dil1.SetKernelRadius(i+1)
		im_dil = dil1.Execute(im)
		#	bin_dil.SetKernelRadius(i)
		#	im_dil = bin_dil.Execute(im)
		#	im_dil = im_dil*i
		if i ==1: 
			im_combined = im_dil
		else:
			im_combined_loc = im_combined>=1
			im_dil_loc = im_dil>=1
			xor_imcombined = xor.Execute(im_combined_loc, im_dil_loc)
	#		im_combined = cast.Execute(im_combined)*cast.Execute(xor_imcombined)+cast.Execute(im_dil)*cast.Execute(xor_imcombined)+cast.Execute(im_dil)*cast.Execute(im_combined_loc*im_dil_loc)
			im_combined = im_combined*cast.Execute(xor_imcombined)+im_dil*cast.Execute(xor_imcombined)+im_dil*cast.Execute(im_combined_loc*im_dil_loc)

	thicknessmap = (im_combined*2+1)*cast.Execute(in_im)
	return thicknessmap

pixelType = sitk.sitkFloat32

#read in images:
gray_imagefnm = 'Scaphoid_16.mha'
gray_image = sitk.ReadImage(gray_imagefnm, sitk.sitkFloat32)

seg_scaphoidfxfnm = 'Scaphoid16_seg_fx.mha'
seg_scaphoidfx = sitk.ReadImage(seg_scaphoidfxfnm, sitk.sitkFloat32)

#apply threshold to isolate contour of the fracture only: 
seg_fx = sitk.BinaryThreshold(seg_scaphoidfx, lowerThreshold=2, upperThreshold=3, insideValue=127, outsideValue=0)
seg_scaphoid = sitk.BinaryThreshold(seg_scaphoidfx, lowerThreshold=1, upperThreshold=2, insideValue=127, outsideValue=0)

sitk.WriteImage(seg_fx,'Scaphoid_fxonly.mha')
sitk.WriteImage(seg_scaphoid, 'Scaphoid_seg.mha')


#smooth scaphoid contour by eroding then dilating it back:

erode = sitk.BinaryErodeImageFilter()
erode.SetKernelRadius(2)
erode.SetForegroundValue(127)
seg_scaphoid_er = erode.Execute(seg_scaphoid)
dilate = sitk.BinaryDilateImageFilter()
dilate.SetKernelRadius(2)
dilate.SetForegroundValue(127)
seg_scaphoid_smooth = dilate.Execute(seg_scaphoid_er)

sitk.WriteImage(seg_scaphoid_smooth,'mask_scaphoidonly_smooth.mha')

#smooth fracture contour by eroding then dilating it back:
erode_fx = sitk.BinaryErodeImageFilter()
erode_fx.SetKernelRadius(1)
erode_fx.SetForegroundValue(127)
dilate_fx = sitk.BinaryDilateImageFilter()
dilate_fx.SetKernelRadius(1)
dilate_fx.SetForegroundValue(127)
seg_fx_er = erode_fx.Execute(seg_fx)
seg_fx_smooth = dilate_fx.Execute(seg_fx_er)

sitk.WriteImage(seg_fx_smooth,'mask_fxonly_smooth.mha')

#re-combine smoothed scaphoid and fx contour: 
seg_scaphoid_smooth = sitk.BinaryThreshold(seg_scaphoid_smooth, lowerThreshold=126, upperThreshold=150, insideValue=1, outsideValue=0)
seg_fx_smooth = sitk.BinaryThreshold(seg_fx_smooth, lowerThreshold=126, upperThreshold=150, insideValue=1, outsideValue=0)
seg_scaphoidfx_smooth = seg_scaphoid_smooth + seg_fx_smooth
sitk.WriteImage(seg_scaphoidfx_smooth, 'mask_scaphoidfx_smoothed.mha')

#Mask grayscale image with the smoothed scaphoid contour: 
cast = sitk.CastImageFilter()
cast.SetOutputPixelType(sitk.sitkFloat32)
scaphoid_mask = gray_image*cast.Execute(seg_scaphoid_smooth)

sitk.WriteImage(scaphoid_mask, 'scaphoid_grayscale_masked.mha')

#Crop by bounding box: 
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(seg_scaphoid_smooth)
bound = stats.GetBoundingBox(1)
imagesize = seg_scaphoid_smooth.GetSize()
crop = sitk.CropImageFilter()
crop.SetLowerBoundaryCropSize([bound[0], bound[1], bound[2]])
crop.SetUpperBoundaryCropSize([(imagesize[0]-bound[0]-bound[3]), (imagesize[1]-bound[1]-bound[4]), (imagesize[2]-bound[2]-bound[5])])

cropped_grayscale = crop.Execute(scaphoid_mask)
cropped_seg = crop.Execute(seg_scaphoid_smooth)
cropped_fx = crop.Execute(seg_fx_smooth)
sitk.WriteImage(cropped_grayscale, 'grayscale_scaphoidonly_cropped.mha')
sitk.WriteImage(cropped_fx, 'mask_fx_cropped.mha')

#Get thickness map of fx site: 
fx_thicknessmap = GetThicknessMap(cropped_fx)	
sitk.WriteImage(fx_thicknessmap, 'fx_thicknessmap.mha')
cc = sitk.ConnectedComponentImageFilter()
fx_label = cc.Execute(cropped_fx)

stats3 = sitk.LabelStatisticsImageFilter()
stats3.Execute(fx_thicknessmap, fx_label)
max_fx_thickness = stats3.GetMaximum(1)
print(max_fx_thickness)
mean_fx_thickness = stats3.GetMean(1)
print(mean_fx_thickness)
std_fx_thickness = stats3.GetSigma(1)
min_fx_thickness = stats3.GetMinimum(1)
print(min_fx_thickness)


#Threshold the masked scaphoid, invert, and compute thickness map of gaps within microstructure
#to get map of "regions of microstructural deterioration"
seg_holes = sitk.BinaryThreshold(cropped_grayscale, lowerThreshold=-1000, upperThreshold=450, insideValue=1, outsideValue=0)
#seg_holes = sitk.BinaryThreshold(cropped_grayscale, upperThreshold=450, insideValue=1, outsideValue=0)

seg_holes = seg_holes*cropped_seg
sitk.WriteImage(seg_holes*127, 'scaphoid_seg_holes.mha')

holes_thicknessmap = GetThicknessMap(seg_holes)
sitk.WriteImage(holes_thicknessmap, 'holes_thickness.mha')

mu_det = sitk.BinaryThreshold(holes_thicknessmap, lowerThreshold=3.5, upperThreshold=100, insideValue=1, outsideValue=0)
fillholes = sitk.BinaryFillholeImageFilter()
mu_det = fillholes.Execute(mu_det)
sitk.WriteImage(mu_det*127,'microstructural_deterioration_map.mha')

#combine fracture, scaphoid, and deterioration masks for visualization in ITK-SNAP
#scaphoid = 1, fracture = 2, deterioration = 3
mask_combined = cast.Execute(cropped_seg) + cast.Execute(cropped_fx)
xor1 = sitk.XorImageFilter()
fxORdet = xor1.Execute(cropped_fx, mu_det)
mask_combined = mask_combined + cast.Execute(mu_det*fxORdet)*2
sitk.WriteImage(mask_combined,'full_labels.mha')
