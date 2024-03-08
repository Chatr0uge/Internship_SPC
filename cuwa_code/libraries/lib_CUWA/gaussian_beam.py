import numpy as np

class GaussianBeam:
    def __init__(self, frequency_GHz = None, mode = None, origin = None, gaussian_focus = None, direction=1., waist_radius = None, E_lab = None, pulse = -1):    
        self.frequency_GHz = frequency_GHz
        self.mode          = mode
        self.origin        = origin
        self.direction     = direction
        self.focus_location= gaussian_focus
        self.Gaussian_focus= np.linalg.norm(self.focus_location - self.origin)
        self.k_lab         = self.direction*(self.focus_location - self.origin)/np.abs(self.Gaussian_focus)
        
        self.w_0            = 2.0*np.pi*1.E9 * frequency_GHz
        self.wavelength     = 2.0*np.pi*299792458.0/self.w_0
        self.Waist_radius   = waist_radius
        self.Rayleigh_range = np.pi * self.Waist_radius**2 / self.wavelength
        
        self.E_lab          = E_lab
        self.pulse = pulse*1E-9 #pulse length for pulse operation. negative value - CW

        #self.radius_at_origin    = self.Waist_radius * np.sqrt(1. + (self.Gaussian_focus/self.Rayleigh_range)**2)
        #self.curvature_at_origin = self.Gaussian_focus * (1. + (self.Rayleigh_range/self.Gaussian_focus)**2)
        
    @classmethod
    def from_TRAVIS_xml_dict(cls,b):
        frequency_GHz  = b['frequency_GHz'] 
        w_0            = 2.0*np.pi*1.E9 * frequency_GHz
        wavelength     = 2.0*np.pi*299792458.0/w_0
        Waist_radius   = b['Radius'] / np.sqrt((np.pi*b['Radius']**2/(wavelength*b['Curvature']))**2+1.0)
        Rayleigh_range = np.pi * Waist_radius**2 / wavelength
        Gaussian_focus = np.sign(b['Curvature'])*Rayleigh_range*np.sqrt((b['Radius']/Waist_radius)**2 - 1.0)
        
        direction      = beam_direction_W7X_angles(b)
        gaussian_focus = b['origin'] + direction * Gaussian_focus
        return cls(frequency_GHz  = frequency_GHz, mode = b['mode'], origin = b['origin'], 
                   gaussian_focus = gaussian_focus, direction=np.sign(b['Curvature']), waist_radius = Waist_radius)


def beam_direction_W7X_angles(beam_config,shift=np.array((0,0))):
    beam_config['direction'][0:2] += np.array(shift)
    direction = beam_config['direction']
    if beam_config['targetType'] == "W7X aiming angles": #not all targetTypes are implemented, please add... 
        alpha, beta   = direction[0:2]/180.*np.pi
        direction     = W7X_aiming_to_Cart(alpha, beta, beam_config['origin'])
    elif beam_config['targetType'] == "cartesian coordinates of target":
        direction -= beam_config['origin']
        direction /= np.linalg.norm(direction)
    else:
        raise Exception("Unknown aiming coordinates (targetType)")
    return direction

def W7X_aiming_to_Cart(alpha, beta, origin):
    direction_cyl = np.array((-np.cos(alpha)*np.cos(beta),np.cos(alpha)*np.sin(beta),np.sin(alpha)))
    phy           = np.arctan2(origin[1],origin[0])
    X             = direction_cyl[0]*np.cos(phy) - direction_cyl[1]*np.sin(phy)
    Y             = direction_cyl[0]*np.sin(phy) + direction_cyl[1]*np.cos(phy)
    return np.array((X,Y,direction_cyl[2]))

class GaussianBeam_old:
    def __init__(self, frequency_GHz = None, mode = None, k_lab = None, Radius = None, Curvature = None):    
        self.frequency_GHz = frequency_GHz
        self.Radius        = Radius
        self.Curvature     = Curvature
        self.mode          = mode
        self.k_lab         = k_lab
        
        self.w_0            = 2.0*np.pi*1.E9 * frequency_GHz
        self.wavelength     = 2.0*np.pi*299792458.0/self.w_0
        self.Waist_radius   = self.Radius / np.sqrt((np.pi*self.Radius**2/(self.wavelength*self.Curvature))**2+1.0)
        self.Rayleigh_range = np.pi * self.Waist_radius**2 / self.wavelength
        self.Gaussian_focus = -np.sign(self.Curvature)*self.Rayleigh_range*np.sqrt((self.Radius/self.Waist_radius)**2 - 1.0)
    
    def moveZ():
        b['Gaussian focus'] += moveZ
        b['Radius'] = b['Waist radius'] * np.sqrt(1. + (b['Gaussian focus']/b['Rayleigh range'])**2)
        b['Curvature'] = (-b['Gaussian focus']) * (1. + (b['Rayleigh range']/b['Gaussian focus'])**2)
