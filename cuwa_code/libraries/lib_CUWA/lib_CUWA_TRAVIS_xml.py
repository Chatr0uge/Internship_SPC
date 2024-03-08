import numpy as np
def read_plasma_TRAVIS_xml(input_data_xml):
    import xml.etree.ElementTree as ET
    
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
        
    return mconf_config, ne_config, te_config

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
    
    beam_config['w_0'] = beam_config['frequency_GHz']*1.E9*2.0*np.pi
    beam_config['wavelength'] =  2.0*np.pi*299792458.0/beam_config['w_0']
    if not int(Beam.find('origin').get('cartesianCoordinates')):        
        XY            = beam_config['origin'][0]*np.exp(1j*beam_config['origin'][1]/180.*np.pi)
        beam_config['origin'][0:2]   = np.array((XY.real,XY.imag))    
    
    beam_modify_W7X_angles(beam_config,np.array((0,0))) #run with (0,0), i.e. no modification to append beam with xyz direction 
    return beam_config

def W7X_aiming_to_Cart(alpha, beta, origin):
    direction_cyl = np.array((-np.cos(alpha)*np.cos(beta),np.cos(alpha)*np.sin(beta),np.sin(alpha)))
    phy           = np.arctan2(origin[1],origin[0])
    X             = direction_cyl[0]*np.cos(phy) - direction_cyl[1]*np.sin(phy)
    Y             = direction_cyl[0]*np.sin(phy) + direction_cyl[1]*np.cos(phy)
    return np.array((X,Y,direction_cyl[2]))

def beam_modify_W7X_angles(beam_config,shift=np.array((0,0))):
    beam_config['direction'][0:2] += np.array(shift)
    direction = beam_config['direction']
    if beam_config['targetType'] == "W7X aiming angles": #I am lazy to write conversions for other targetTypes, please add... 
        alpha, beta   = direction[0:2]/180.*np.pi
        direction     = W7X_aiming_to_Cart(alpha, beta, beam_config['origin'])
    elif beam_config['targetType'] == "cartesian coordinates of target":
        direction -= beam_config['origin']
        direction /= np.linalg.norm(direction)
    else:
        raise Exception("Unknown aiming coordinates (targetType)")
    beam_config['direction_XYZ'] = direction
    
def TRAVIS_profiles(ne_c, te_c):        
    ne     = lambda _x:  ne_c['Ne0']*(ne_c['g']-ne_c['hole']+
                 (1.-ne_c['g']+ne_c['hole'])*np.fmax(0.0,(1.-_x**(0.5*ne_c['p'])))**ne_c['q']+ 
                     ne_c['hole']*(1.-np.exp(-_x/ne_c['width']**2)))

    dne_ds = lambda _x: -ne_c['Ne0']*(1.-ne_c['g']+ne_c['hole'])*0.5*ne_c['p']*ne_c['q']*_x**(0.5*ne_c['p']-1)*(1.-_x**(0.5*ne_c['p']))**(ne_c['q']-1)+ne_c['Ne0']*ne_c['hole']*np.exp(-_x/ne_c['width']**2)/ne_c['width']**2  

    te   = lambda _x: te_c['Te0']*(te_c['g']-te_c['hole']+
              (1.-te_c['g']+te_c['hole'])*np.fmax(0.0,(1.-_x**(0.5*te_c['p'])))**te_c['q']+
                  te_c['hole']*(1.-np.exp(-_x/te_c['width']**2)))
    return ne, dne_ds, te
