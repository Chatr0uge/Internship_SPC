import numpy as np
import sys,os
import xml.etree.ElementTree as ET

def Write_XMF(file,S_list,timed=False):
    XMFtext=\
"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
  <Domain>
"""
    if timed:
        XMFtext+="""
    <Grid Name="Time" GridType="Collection" CollectionType="Temporal">
"""
    if isinstance(S_list[0], list):
        for block in S_list:
            field_name,file_name,time,shape,dx,OriginZ,OriginY,OriginX = block
            Size_Z,Size_Y,Size_X = shape
            XMFtext+=XMF_slice_body(Size_Z,Size_Y,Size_X,dx,-OriginZ,-OriginY,-OriginX,time,field_name,file_name)
    else:
        field_name,file_name,time,shape,dx,OriginZ,OriginY,OriginX = S_list
        Size_Z,Size_Y,Size_X = shape
        XMFtext+=XMF_slice_body(Size_Z,Size_Y,Size_X,dx,-OriginZ,-OriginY,-OriginX,time,field_name,file_name)

    if timed:
        XMFtext+="""
    </Grid>
"""
    XMFtext+="""
  </Domain>
</Xdmf>
"""
    text_file = open(file, "w")
    text_file.write(XMFtext)
    text_file.close()

def XMF_slice_body(Size_Z,Size_Y,Size_X,dx,OriginZ,OriginY,OriginX,time,field_name,file_name):
    return """
      <Grid Name="Structured Grid" GridType="Uniform">
        <Time Value="{7}" />
        <Topology TopologyType="3DCORECTMesh" Dimensions="{0} {1} {2}"/>
        <Geometry GeometryType="ORIGIN_DXDYDZ">
          <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">
              {4} {5} {6}
          </DataItem>
          <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">
              {3} {3} {3}
          </DataItem>
        </Geometry>
        <Attribute Type="Vector" Center="Node" Name="{8}">
            <DataItem ItemType="Function" Function="JOIN($0, $1, $2)" Dimensions="{0} {1} {2} 3">
                 <DataItem Name="{8}x" Format="HDF" DataType="Float" Dimensions="{0} {1} {2}">
                   {9}.h5:/{8}x                                                             
                 </DataItem>                                                              
                 <DataItem Name="{8}y" Format="HDF" DataType="Float" Dimensions="{0} {1} {2}">
                   {9}.h5:/{8}y                                                             
                 </DataItem>                                                              
                 <DataItem Name="{8}z" Format="HDF" DataType="Float" Dimensions="{0} {1} {2}">
                   {9}.h5:/{8}z
            </DataItem>
          </DataItem>
        </Attribute>
      </Grid>
""".format(Size_Z,Size_Y,Size_X,dx,OriginZ,OriginY,OriginX,time,field_name,file_name)

def Write_VTP(file,rays_list):
    XMFtext=\
"""<?xml version="1.0" ?>
<VTKFile type="PolyData" version="0.1">
        <PolyData>
"""
    for ray in rays_list:
        XMFtext+=VTP_slice_body(ray)
    XMFtext+="""
        </PolyData>
</VTKFile>
"""
    text_file = open(file, "w")
    text_file.write(XMFtext)
    text_file.close()

def VTP_slice_body(ray):
    return """
                <Piece NumberOfPoints="{0}" NumberOfLines="1" NumberOfPolys="1">
                        <Points>
                                <DataArray type="Float32" NumberOfComponents="3" format="ascii">
                                         {2}
                                </DataArray>
                        </Points>
                        <Lines>
                                <DataArray type="Int32" Name="connectivity" format="ascii">
                                         {1}
                                </DataArray>
                                <DataArray type="Int32" Name="offsets" format="ascii">
                                         {0}
                                </DataArray>
                        </Lines>
                </Piece>
""".format(ray[0].size,' '.join(str(x) for x in range(0,ray[0].size)),'\n'.join(' '.join(str(cell) for cell in row) for row in np.transpose([ray[0],ray[1],ray[2]])))

def h5_out(file_name,h5_output,dx,origin):
    import h5py
    with h5py.File(file_name, "w") as hf:
        for field in h5_output.keys():
            dset = hf.create_dataset(field, data=h5_output[field])
        hf.close()
    Write_scalars_XMF(file_name,h5_output.keys(),
                               list(h5_output.values())[0].shape,dx,origin[0],origin[1],origin[2])

def Write_scalars_XMF(file_name,names,shape,dx,OriginZ,OriginY,OriginX):
    XMFtext=\
"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
 <Domain>
   <Grid Name="Structured Grid" GridType="Uniform">
     <Topology TopologyType="3DCORECTMesh" Dimensions="{0} {1} {2}"/>
     <Geometry GeometryType="ORIGIN_DXDYDZ">
       <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">
           {4} {5} {6}
       </DataItem>
       <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">
           {3} {3} {3}
       </DataItem>
     </Geometry>""".format(shape[0],shape[1],shape[2],dx,-OriginZ,-OriginY,-OriginX)
    
    for field in names:
        XMFtext+="""
     <Attribute Type="Scalar" Center="Node" Name="{3}">
              <DataItem Format="HDF" DataType="Float" Dimensions="{0} {1} {2}">
                {4}:/{3}
              </DataItem>
     </Attribute>""".format(shape[0],shape[1],shape[2],field,os.path.split(file_name)[1])
    
    XMFtext+="""   
    </Grid>
 </Domain>
</Xdmf>
    """
    text_file = open(os.path.splitext(file_name)[0]+".xmf", "w")
    text_file.write(XMFtext)
    text_file.close()

def read_TRAVIS_xml(input_data_xml):
    """ Reads TRAVIS xml file.
    returns dicts: 
    beam_names, beams_data, mconf_config, ne_config, te_config, smooth_config
    """
    beam_names = {}
    for beam_element in ET.parse(input_data_xml).getroot().findall('ECRHsystem/Beam'):
        beam_names[int(beam_element.get('id'))] = beam_element.get('name')
        
    beams_data = {}   
    for beam_id in beam_names.keys():
        beams_data[beam_id] = read_beam_TRAVIS_xml(input_data_xml, beam_id)
    
    mconf_config, ne_config, te_config, smooth_config = read_plasma_TRAVIS_xml(input_data_xml)
    return beam_names, beams_data, mconf_config, ne_config, te_config, smooth_config
    
def read_plasma_TRAVIS_xml(input_data_xml):
    
    mconf_xml  = ET.parse(input_data_xml).getroot()
    mconf_config={
           'eq_file'    :mconf_xml.find("MagneticConfig").get('name'),
           'B0'         :float(mconf_xml.find("MagneticConfig/B0").text) if not mconf_xml.find("MagneticConfig/B0").get('type') =="don't scale" else None,
           'B0_angle'   :float(mconf_xml.find("MagneticConfig/B0").get('angle')),
           'step'       :float(mconf_xml.find("MagneticConfig/Mesh/step").text),
           'angleStep'  :float(mconf_xml.find("MagneticConfig/Mesh/angleStep").text),
           'accuracy'   :float(mconf_xml.find("MagneticConfig/Mesh/accuracy").text),
           'truncation' :float(mconf_xml.find("MagneticConfig/Mesh/truncation").text),
           'extraLCMS'  :float(mconf_xml.find("MagneticConfig/Mesh/extraLCMS").text)}
    
    Ne  = ET.parse(input_data_xml).getroot().find("Ne")
    ne_config = {key:float(Ne.get(key))  for key in ['Ne0','g','p','q','hole','width']}

    Te   = ET.parse(input_data_xml).getroot().find("Te")
    te_config = {key:float(Te.get(key))  for key in ['Te0','g','p','q','hole','width']}
        
    PlasmaSize = ET.parse(input_data_xml).getroot().find("PlasmaSize")
    smooth_config = None
    if PlasmaSize is not None:
        smooth_config = {key:float(PlasmaSize.get(key))  for key in ['rhoMax','edgeWidth']}
    
    return mconf_config, ne_config, te_config, smooth_config

def read_beam_TRAVIS_xml(input_data_xml, beam_id=None):
    if beam_id is None:
        print('beam_id argument is needed')
        return 
    
    Beam          = ET.parse(input_data_xml).getroot().find('ECRHsystem/Beam[@id="{}"]'.format(beam_id))
    beam_config = {\
            'name'                  : Beam.attrib['name'],
            'frequency_GHz'         : np.float(Beam.find('frequency'  ).text.split()[0]), #GHz
            'mode'                  :          Beam.find('heatingMode').text.split()[0],  #X or O mode
            'Radius'                : np.float(Beam.find('radius').text.split()[0]), #m
            'Curvature'             : np.float(Beam.find('focal').text.split()[0]),  #m NOTE: this focal is GO focus in TRAVIS
            'origin'                : np.array(Beam.find('origin').text.split(),dtype=np.float),    #m
            'direction'             : np.array(Beam.find('direction').text.split(),dtype=np.float),
            'targetType'            : Beam.find('direction').get('targetType')}     #m
    
    #beam_config['w_0'] = beam_config['frequency_GHz']*1.E9*2.0*np.pi
    #beam_config['wavelength'] =  2.0*np.pi*299792458.0/beam_config['w_0']
    if not int(Beam.find('origin').get('cartesianCoordinates')):        
        XY            = beam_config['origin'][0]*np.exp(1j*beam_config['origin'][1]/180.*np.pi)
        beam_config['origin'][0:2]   = np.array((XY.real,XY.imag))    
    
    #beam_modify_W7X_angles(beam_config,np.array((0,0))) #run with (0,0), i.e. no modification to append beam with xyz direction 
    return beam_config
    
def TRAVIS_profiles(ne_c, te_c, s_c = None):
    
    smooth = lambda _x, s1, s2: 0.5+0.5*np.cos( np.pi*np.clip((_x-s1)/(s2-s1)) )
    
    ne     = lambda _x:  1e20*ne_c['Ne0']*(ne_c['g']-ne_c['hole']+
                 (1.-ne_c['g']+ne_c['hole'])*np.fmax(0.0,(1.-_x**(0.5*ne_c['p'])))**ne_c['q']+ 
                     ne_c['hole']*(1.-np.exp(-_x/ne_c['width']**2)))

    dne_ds = lambda _x: -1.e20*ne_c['Ne0']*(1.-ne_c['g']+ne_c['hole'])*((1.-_x**(0.5*ne_c['p']))>0)*0.5*ne_c['p']*ne_c['q']*_x**(0.5*ne_c['p']-1)*(1.-_x**(0.5*ne_c['p']))**(ne_c['q']-1.)+1e20*ne_c['Ne0']*ne_c['hole']*np.exp(-_x/ne_c['width']**2)/ne_c['width']**2  

    te   = lambda _x: te_c['Te0']*(te_c['g']-te_c['hole']+
              (1.-te_c['g']+te_c['hole'])*np.fmax(0.0,(1.-_x**(0.5*te_c['p'])))**te_c['q']+
                  te_c['hole']*(1.-np.exp(-_x/te_c['width']**2)))
    return ne, dne_ds, te

