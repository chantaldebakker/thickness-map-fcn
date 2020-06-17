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
