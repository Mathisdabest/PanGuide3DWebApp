import zipfile
import numpy as np
import pytest
import SimpleITK as sitk
from webcore import validate_image, load_upload, restore, statistics, report_text, slices, mesh, export_pdf, dicom_image
from panguide3d.preprocess import preprocess_array, window_scale

def scan():
    a=np.full((12,16,20),-1000,dtype=np.float32); a[3:9,4:12,5:15]=50
    im=sitk.GetImageFromArray(a); im.SetSpacing((1,2,3)); im.SetOrigin((11,22,33))
    return im

def test_window():
    assert np.allclose(window_scale(np.array([-200,-125,75,275,500])),[0,0,.5,1,1])

def test_nifti_geometry_and_restore(tmp_path):
    im=scan(); im.SetDirection((-1,0,0,0,1,0,0,0,-1))
    path=tmp_path/'ct.nii.gz'; sitk.WriteImage(im,str(path))
    original,vol=load_upload(path)
    restored=restore(np.ones(vol.shape,dtype=np.uint8),vol,original)
    assert restored.GetSize()==im.GetSize()
    assert restored.GetOrigin()==im.GetOrigin()
    assert restored.GetDirection()==im.GetDirection()
    assert restored.GetSpacing()==im.GetSpacing()
    # A physical landmark survives both grids within one output voxel.
    landmark=sitk.Image(im.GetSize(),sitk.sitkUInt8); landmark.CopyInformation(im)
    arr=sitk.GetArrayFromImage(landmark); arr[4:8,6:10,7:13]=1
    landmark=sitk.GetImageFromArray(arr); landmark.CopyInformation(im)
    target=sitk.GetImageFromArray(np.zeros(vol.shape,dtype=np.uint8))
    target.SetSpacing(vol.spacing); target.SetOrigin(vol.origin); target.SetDirection(vol.direction)
    modelmask=sitk.Resample(landmark,target,sitk.Transform(),sitk.sitkNearestNeighbor)
    back=sitk.GetArrayFromImage(restore(sitk.GetArrayFromImage(modelmask),vol,im))
    assert np.linalg.norm(np.argwhere(back).mean(0)-np.argwhere(arr).mean(0))<1.5

def test_invalid_and_non_ct(tmp_path):
    im=sitk.GetImageFromArray(np.ones((8,8,8),dtype=np.float32))
    with pytest.raises(ValueError,match='CT intensity'): validate_image(im)
    im=sitk.GetImageFromArray(np.ones((8,8),dtype=np.float32))
    with pytest.raises(ValueError,match='3D'): validate_image(im)
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z: z.writestr('../evil','x')
    with pytest.raises(ValueError,match='Unsafe'): load_upload(p)

def dataset():
    ct=sitk.GetArrayFromImage(scan()); pan=np.zeros(ct.shape,dtype=np.uint8); pan[3:9,4:12,5:15]=1
    tumor=np.zeros_like(pan); tumor[4:6,5:7,6:8]=1
    return dict(ct=ct,pan=pan,tumor=tumor,spacing=(1,2,3),origin=(11,22,33),direction=(1,0,0,0,1,0,0,0,1),provenance='test fixture, not model output',elapsed=0)

def test_intersection_and_report():
    d=dataset(); s=statistics(d['pan'],d['tumor'],d['spacing'])
    assert s['tumor_cm3']==pytest.approx(.048)
    assert s['intersection_cm3']==s['tumor_cm3']
    assert s['tumor_inside_pancreas_percent']==100
    d['tumor'][:]=0
    assert 'does not establish the absence' in report_text(d)

def test_render_export(tmp_path):
    d=dataset(); images=slices(d,5,7,10,400,50,.5,True,True)
    assert len(images)==3
    fig=mesh(d); assert len(fig.data)==2
    pdf=export_pdf(d,images,tmp_path)
    assert open(pdf,'rb').read(4)==b'%PDF'

def make_dicom(tmp_path,index,uid='1.2.3'):
    import pydicom
    from pydicom.dataset import FileDataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,CTImageStorage,generate_uid
    meta=FileMetaDataset(); meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=CTImageStorage; meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=CTImageStorage; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID=uid; ds.StudyInstanceUID='1.2.4'; ds.FrameOfReferenceUID='1.2.5'
    ds.Modality='CT'; ds.PatientName='TEST^IDENTIFIER'; ds.PatientID='fixture'
    ds.Rows=8; ds.Columns=8; ds.PixelSpacing=[1,2]; ds.ImageOrientationPatient=[1,0,0,0,1,0]
    ds.ImagePositionPatient=[0,0,index*3]; ds.InstanceNumber=index+1
    ds.SamplesPerPixel=1; ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=1
    ds.RescaleSlope=2; ds.RescaleIntercept=-1000
    ds.PixelData=np.full((8,8),index,dtype=np.int16).tobytes()
    ds.is_little_endian=True; ds.is_implicit_VR=False
    path=tmp_path/f'{index}-{uid}.dcm'; ds.save_as(str(path),write_like_original=False)
    return path

def test_dicom_sort_rescale_and_no_identifiers(tmp_path):
    files=[make_dicom(tmp_path,i) for i in (2,0,1)]
    im=dicom_image(files)
    assert im.GetSpacing()==(2,1,3)
    assert np.allclose(sitk.GetArrayFromImage(im)[:,0,0],[-1000,-998,-996])
    assert not im.HasMetaDataKey('0010|0010')
    files.append(make_dicom(tmp_path,3,'1.2.99'))
    with pytest.raises(ValueError,match='coherent'): dicom_image(files)

def test_checkpoint_integrity(tmp_path):
    from panguide3d.model import load_model
    p=tmp_path/'bad.pt'; p.write_bytes(b'untrusted pickle')
    with pytest.raises(ValueError,match='SHA-256'): load_model(p,'cpu')


def test_evidence_is_not_a_diagnosis_probability():
    from webcore import reliability_text,findings_summary
    text=reliability_text()
    assert 'not established' in text
    assert '0.460 ± 0.041' in text and '0.475 ± 0.069' in text
    assert 'not validated performance of this MSD-trained checkpoint' in text
    assert 'not an accuracy score' in text
    d=dataset(); text=findings_summary(d)
    assert '0.05 cm³' in text and '100.0%' in text
    assert 'not confirmed cancer' in text
    d['pan'][:]=0
    assert 'localization is unreliable' in findings_summary(d)

def test_intersection_is_a_real_mask_surface():
    d=dataset(); f=mesh(d)
    assert 'Mask intersection (not diagnostic)' in [t.name for t in f.data]
    assert len(mesh(d,show_pan=False).data)==1
    d['tumor'][:]=0
    assert len(mesh(d).data)==1


def test_confidence_uses_actual_voxel_probabilities():
    from webcore import confidence_summary,confidence_text
    mask=np.zeros((4,4,4),dtype=np.uint8);mask[0,0,:2]=1;mask[3,3,3]=1
    prob=np.zeros_like(mask,dtype=float);prob[0,0,:2]=[.8,1.0];prob[3,3,3]=.6
    scores=confidence_summary(mask,prob)
    assert np.allclose(scores['tumor_confidence_means'],[.9,.6])
    assert list(scores['tumor_confidence_voxels'])==[2,1]
    text=confidence_text(dict(tumor=mask,**scores))
    assert '80.0%' in text and '90.0%' in text and '60.0%' in text
    assert 'not the probability that the region is cancerous' in text
    assert 'Unavailable' in confidence_text(dict(tumor=mask))
    assert 'Not applicable' in confidence_text(dict(tumor=np.zeros_like(mask)))
    with pytest.raises(ValueError):confidence_summary(mask,prob[:2])
    prob[0,0,0]=float('nan')
    with pytest.raises(ValueError):confidence_summary(mask,prob)
