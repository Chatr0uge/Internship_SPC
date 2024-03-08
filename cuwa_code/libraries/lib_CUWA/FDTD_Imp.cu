#include <stdio.h>

#define CUDART_PI_F 3.141592654f


//#define _KAPPA     5.f//5.0f
//#define _ALPHA     0.5f//1f*0.025f //    alpha0 * (dt/epsilon0)
//#define _M1        2
//#define _MA        2
//#define _SIGMA_MAX 1.f//0.080f

//#define x_width    640
//#define y_width    640
//#define z_width    8
//#define _nPML      10
//#define _Courant   0.5f
//#define _grid_scale 8
#define _Te_const 1

#define _Full2D 1
//#define _J_intrpolate 1 //without interpolation it is advisible to use lengthy wp rampdown in set_plasma_data. Otherwise CPML unstable. Energy conservation is better without the interpolation. _J_intrpolate = 2 is interpolation from Inan.
//#define _Jdt 2 // "3" (fully implicit E-J in time) is not implemented for 3D

struct kappa_a_b
{
    float kappa; 
    float a;
    float b;
};

__device__ kappa_a_b CPML_grading(const float l)
{
    float sigma_max = _SIGMA_MAX * (_M1+1) * 0.8f * fabsf(_Courant);
    float l_PML     = powf((__int2float_rn(_nPML) - l) / (__int2float_rn(_nPML)),_M1);
    float sigma     = sigma_max * l_PML; // Eq. 9.30a
    float alpha     = _ALPHA * powf((l)/(__int2float_rn(_nPML)),_MA);// powf(idx/(nyPML_1-1.0),ma=1)
    float kappa     = 1.f+(_KAPPA-1.f) * l_PML; //Eq. 9.30b
    float b         = expf(-(sigma / kappa + alpha)); //Eq. 9.95c
    float a         = (b-1.f) / ((1.f+ kappa * alpha / sigma) * kappa);//Eq. 9.92
    kappa_a_b v     = {kappa,a,b};
    return v;
}

__device__ float CPML_kappa(const float l)
{
    float l_PML     = powf((__int2float_rn(_nPML) - l) / (__int2float_rn(_nPML)),_M1);
    float kappa     = 1.f+(_KAPPA-1.f) * l_PML; //Eq. 9.30b
    return kappa;
}

typedef struct Matrix{
        float e11,e12,e13;
        float e21,e22,e23;
        float e31,e32,e33;
        
        __device__ Matrix(): 
                            e11(0.f), e12(0.f), e13(0.f),
                            e21(0.f), e22(0.f), e23(0.f),
                            e31(0.f), e32(0.f), e33(0.f) {};
        
        __device__ Matrix(float _e11,float _e12,float _e13,
                          float _e21,float _e22,float _e23,
                          float _e31,float _e32,float _e33): 
                            e11(_e11), e12(_e12), e13(_e13),
                            e21(_e21), e22(_e22), e23(_e23),
                            e31(_e31), e32(_e32), e33(_e33) {};

        __device__ Matrix operator=(const Matrix a) {
            return Matrix(e11=a.e11, e12=a.e12, e13=a.e13,
                          e21=a.e21, e22=a.e22, e23=a.e23,
                          e31=a.e31, e32=a.e32, e33=a.e33);
        }

        __device__ Matrix operator*(const Matrix &a) {
            Matrix result(e11 * a.e11 + e12 * a.e21 + e13 * a.e31, e11 * a.e12 + e12 * a.e22 + e13 * a.e32, e11 * a.e13 + e12 * a.e23 + e13 * a.e33,
                          e21 * a.e11 + e22 * a.e21 + e23 * a.e31, e21 * a.e12 + e22 * a.e22 + e23 * a.e32, e21 * a.e13 + e22 * a.e23 + e23 * a.e33,
                          e31 * a.e11 + e32 * a.e21 + e33 * a.e31, e31 * a.e12 + e32 * a.e22 + e33 * a.e32, e31 * a.e13 + e32 * a.e23 + e33 * a.e33);
            return result;
        }

        __device__ Matrix operator+(const Matrix &a) {
            Matrix result(e11 + a.e11, e12 + a.e12, e13 + a.e13,
                          e21 + a.e21, e22 + a.e22, e23 + a.e23,
                          e31 + a.e31, e32 + a.e32, e33 + a.e33);
            return result;
        }

        __device__ Matrix operator-(const Matrix &a) {
            Matrix result(e11 - a.e11, e12 - a.e12, e13 - a.e13,
                          e21 - a.e21, e22 - a.e22, e23 - a.e23,
                          e31 - a.e31, e32 - a.e32, e33 - a.e33);
            return result;
        }

        __device__ Matrix operator*(const float &a) {
            Matrix result(a*e11, a*e12, a*e13,
                          a*e21, a*e22, a*e23,
                          a*e31, a*e32, a*e33);
            return result;
        }

        __device__ friend Matrix operator*(const float &a, const Matrix &b) {
            Matrix result(a*b.e11, a*b.e12, a*b.e13,
                          a*b.e21, a*b.e22, a*b.e23,
                          a*b.e31, a*b.e32, a*b.e33);
            return result;
        }

        __device__ Matrix operator/(const float &a) {
            Matrix result(e11/a, e12/a, e13/a,
                          e21/a, e22/a, e23/a,
                          e31/a, e32/a, e33/a);
            return result;
        }

        __device__ float det(void){
            return e11 * (e22 * e33 - e23 * e32)
                  -e12 * (e21 * e33 - e23 * e31)
                  +e13 * (e21 * e32 - e22 * e31);
        }

        __device__ Matrix inv(void){
            Matrix result(  e22 * e33 - e23 * e32 , -(e12 * e33 - e13 * e32),  e12 * e23 - e13 * e22 ,
                          -(e21 * e33 - e23 * e31),   e11 * e33 - e13 * e31 ,-(e11 * e23 - e13 * e21),
                            e21 * e32 - e22 * e31 , -(e11 * e32 - e12 * e31),  e11 * e22 - e12 * e21 );
            return result/(e11 * (e22 * e33 - e23 * e32)-e12 * (e21 * e33 - e23 * e31)+e13 * (e21 * e32 - e22 * e31));
        }
        
} Matrix;

__global__ void update_3D_H_CPML( 
                                 const float * __restrict__ Ex_g, 
                                 const float * __restrict__ Ey_g, 
                                 const float * __restrict__ Ez_g, 
                                       float * __restrict__ Hx_g, 
                                       float * __restrict__ Hy_g, 
                                       float * __restrict__ Hz_g, 
                                       float * __restrict__ pHyx_g,
                                       float * __restrict__ pHzx_g,
                                       float * __restrict__ pHxy_g,
                                       float * __restrict__ pHzy_g)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)

    //int x_width = blockDim.x * gridDim.x;
    //int y_width = blockDim.y * gridDim.y;
    
    #if _3D
    for(int idz = 0; idz < z_width - 1; idz++) // loop over z coordinate (numpy axis 0)
    #endif
    {
        #if _3D
            int ijk = idx +     x_width * idy     + (x_width * y_width) *  idz;
            int i1jk= idx + 1 + x_width * idy     + (x_width * y_width) *  idz;
            int ij1k= idx +     x_width *(idy + 1)+ (x_width * y_width) *  idz;
            int ijk1= idx +     x_width * idy     + (x_width * y_width) *  (idz + 1);
        #else
            int ijk = idx +     x_width * idy;
            int i1jk= idx + 1 + x_width * idy;
            int ij1k= idx +     x_width *(idy + 1);
        #endif

        //if((idx < x_width - 1) && (idy < y_width - 1)){
            float Ex  =  Ex_g[ijk];
            float Ey  =  Ey_g[ijk];
            #if _Full2D
            float Ez  =  Ez_g[ijk];
            float pHyx = 0.f;
            float pHxy = 0.f;
            #endif

            float pHzx = 0.f;
            float pHzy = 0.f;

            float kappa_x = 1.f;
            float kappa_y = 1.f;
            if(((idx < _nPML) || (idx >= x_width - 1 - _nPML))){
                float l = __int2float_rn(idx) + 0.5f;
                #if _3D
                    int ijk_x1PML = idx +     _nPML * idy   + (2 * _nPML * y_width) *  idz;
                #else
                    int ijk_x1PML = idx +     _nPML * idy;
                #endif
                if (idx >= x_width - 1 - _nPML) {
	                l = __int2float_rn(-idx + x_width - 1) - 0.5f;
                    #if _3D
                        ijk_x1PML = (idx - (x_width - 1 - _nPML)) + idy * _nPML + _nPML * y_width + (2 * _nPML * y_width) *  idz;
                    #else
                        ijk_x1PML = (idx - (x_width - 1 - _nPML)) + idy * _nPML + _nPML * y_width;
                    #endif
	            }
                kappa_a_b k_a_b_x = CPML_grading(l);

                #if _Full2D
                if(idx < x_width - 1) 
                    pHyx = k_a_b_x.b * pHyx_g[ijk_x1PML] + k_a_b_x.a * _Courant * ((Ez_g[i1jk] - Ez));
                pHyx_g[ijk_x1PML] = pHyx;
                #endif
                if ((idx < x_width - 1) && (idy < y_width - 1))
                    pHzx = k_a_b_x.b * pHzx_g[ijk_x1PML] + k_a_b_x.a * _Courant * ((Ey_g[i1jk] - Ey));

                pHzx_g[ijk_x1PML] = pHzx;

                kappa_x=k_a_b_x.kappa;
            } 

            if((idy < _nPML) || (idy >= y_width - 1 - _nPML)){
                float l = __int2float_rn(idy) + 0.5f;
                #if _3D
                    int ijk_y1PML = idx +     x_width * idy  + (2 * _nPML * x_width) *  idz;
                #else
                    int ijk_y1PML = idx +     x_width * idy;
                #endif
                if (idy >= y_width - 1 - _nPML) {
	                l = __int2float_rn(-idy + y_width - 1) - 0.5f;
                    #if _3D
                        ijk_y1PML = idx + (idy - (y_width - 1 - _nPML)) * x_width + _nPML*x_width + (2 * _nPML * x_width) *  idz;
                    #else
                        ijk_y1PML = idx + (idy - (y_width - 1 - _nPML)) * x_width + _nPML*x_width;
                    #endif
	            }
                kappa_a_b k_a_b_y = CPML_grading(l);

                #if _Full2D
                if(idy < y_width - 1)
                    pHxy = k_a_b_y.b * pHxy_g[ijk_y1PML] + k_a_b_y.a * _Courant  * ((Ez_g[ij1k] - Ez));
                pHxy_g[ijk_y1PML]= pHxy;
                #endif
                if ((idx < x_width - 1) && (idy < y_width - 1))
                    pHzy = k_a_b_y.b * pHzy_g[ijk_y1PML] + k_a_b_y.a * _Courant * ((Ex_g[ij1k] - Ex));
                pHzy_g[ijk_y1PML]= pHzy;

                kappa_y=k_a_b_y.kappa;
            } 

            #if _3D
                if (idy < y_width - 1){
                    Hx_g[ijk] += _Courant * ( (Ey_g[ijk1] - Ey)         - (Ez_g[ij1k] - Ez)/kappa_y) - pHxy;
                }
                if (idx < x_width - 1){
                    Hy_g[ijk] += _Courant * ( (Ez_g[i1jk] - Ez)/kappa_x - (Ex_g[ijk1] - Ex)        ) + pHyx;
                }
                if ((idx < x_width - 1) && (idy < y_width - 1)) {
                    Hz_g[ijk] += _Courant * ( (Ex_g[ij1k] - Ex)/kappa_y - (Ey_g[i1jk] - Ey)/kappa_x) + pHzy - pHzx;
                }
            #else
                #if _Full2D
                if (idy < y_width - 1){
                    Hx_g[ijk] += _Courant * (-(Ez_g[ij1k] - Ez)/kappa_y) - pHxy;
                }
                if (idx < x_width - 1){
                    Hy_g[ijk] += _Courant * ( (Ez_g[i1jk] - Ez)/kappa_x) + pHyx;
                }
                #endif
                if ((idx < x_width - 1) && (idy < y_width - 1)) {
                    Hz_g[ijk] += _Courant * ( (Ex_g[ij1k] - Ex)/kappa_y - (Ey_g[i1jk] - Ey)/kappa_x) + pHzy - pHzx;
                }
            #endif
        //}
        //No need for the updates at the edges. These are not updated anyway.
        //if(((idx == 0) || (idx == x_width - 1)) && (idy < y_width - 1)) {
        //	Hx_g[ijk]=0.f;
        //}
        //if(((idy == 0) || (idy == y_width - 1)) && (idx < x_width - 1)) {
        //	Hy_g[ijk]=0.f;
        //}
    }
}

__global__ void update_3D_J_CPML( 
                            const float * __restrict__ Ex_g, 
                            const float * __restrict__ Ey_g, 
                            const float * __restrict__ Ez_g, 
                            const float * __restrict__ Hx_g, 
                            const float * __restrict__ Hy_g, 
                            const float * __restrict__ Hz_g, 
                                  float * __restrict__ Jx_g, 
                                  float * __restrict__ Jy_g, 
                                  float * __restrict__ Jz_g, 
                            const float * __restrict__ Jx0_g, 
                            const float * __restrict__ Jy0_g, 
                            const float * __restrict__ Jz0_g, 
                            const float * __restrict__ bx_g, 
                            const float * __restrict__ by_g, 
                            const float * __restrict__ bz_g, 
                            const float * __restrict__ wp2_g,
                                                 float nudt)
{
    int   idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int   idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    //int   x_width = blockDim.x * gridDim.x;
    //int   y_width = blockDim.y * gridDim.y;

    #if _3D
    for(int idz = 1; idz < z_width - 1; idz++) // loop over z coordinate (numpy axis 0)
    #endif
    {
        if((idx < x_width - 1) && (idy < y_width - 1)){
            #if _3D
                int ijk_8 = idx / _grid_scale +     (x_width / _grid_scale) * (idy / _grid_scale)  + (x_width / _grid_scale) * (y_width / _grid_scale) *  (idz / _grid_scale);
                int ijk = idx +     x_width * idy  + (x_width * y_width) *  idz;
            #else
                int ijk_8 = idx / _grid_scale +     (x_width / _grid_scale) * (idy / _grid_scale);
                int ijk = idx +     x_width * idy;
            #endif

            float bx =  bx_g[ijk_8]; //this is wc_x * dt
            float by =  by_g[ijk_8]; //        wc_y * dt
            float bz =  bz_g[ijk_8]; //        wc_z * dt
            float wp2=  wp2_g[ijk];  //        wp^2 * dt^2
             
            #if _Te_const //nudt
            nudt *= wp2;
            #endif

            Matrix I(1.f,0.f,0.f,                                                              
                     0.f,1.f,0.f,                                                              
                     0.f,0.f,1.f);                                                             
            Matrix JJ;                                                                         
            Matrix JE;                                                                         

            #if _Jdt == 0 
            //Nu = 0 version/////////////////////////////////////////////////////////////////////
            Matrix A0(0.f,-bz,by,                                                              //
                      bz,0.f,-bx,                                                              //
                      -by,bx,0.f);                                                             //
            JJ = (I - A0/2).inv() * (I + A0/2);                                                //
            JE = wp2 * ((I - A0/2).inv());                                                     //
            /////////////////////////////////////////////////////////////////////////////////////
            #elif _Jdt == 1
            //VERSION WITH FIRST ORDER/??////////////////////////////////////////////////////////              
            Matrix A0(-nudt,-bz, by,                                                           //
                       bz,-nudt,-bx,                                                           //
                      -by, bx,-nudt);                                                          //
            JJ = ((I - A0/2).inv()) * (I + A0/2);                                              //
            JE = wp2 * ((I - A0/2).inv());                                                     //
            /////////////////////////////////////////////////////////////////////////////////////
            #elif _Jdt == 2
            //CONSERVATIVE VERSION FROM INAN/////////////////////////////////////////////////////              
            float sin;                                                                         //
            float cos;                                                                         //
            float wcdt = norm3df(bx,by,bz);                                                    // 
            sincosf(wcdt, &sin, &cos);                                                         //
            nudt = max(1.e-12f, nudt);
            float exp = expf(nudt);                                                            //
            Matrix A0(0.f,-bz,by,                                                              //
                      bz,0.f,-bx,                                                              //
                      -by,bx,0.f);                                                             //
            Matrix bb(bx*bx,by*bx,bz*bx,                                                       //
                      bx*by,by*by,bz*by,                                                       //
                      bx*bz,by*bz,bz*bz);                                                      //
            JJ = (cos * I + sin / wcdt * A0 + (1.f - cos) * bb / powf(wcdt,2))/exp;            //
            float C2 = expm1f(nudt) / nudt - nudt * (1.f - cos) / powf(wcdt,2) - sin / wcdt;   //
            float C3 = nudt * (exp - cos) + wcdt * sin;                                        //
            float C4 = exp - cos - nudt * sin / wcdt;                                          //
            JE = wp2 * (C3  * I + C4 * A0 + C2 * bb)/ (exp * (wcdt * wcdt + nudt * nudt)) ;    //
            /////////////////////////////////////////////////////////////////////////////////////
            #elif _Jdt == 3
            //Implicit E-j///////////////////////////////////////////////////////////////////////
            float theta = wp2/4.f*(-expm1f(-nudt)/nudt);                                       //
            Matrix A0(0.f,-bz,by,                                                              //
                      bz,0.f,-bx,                                                              //
                      -by,bx,0.f);                                                             //
            Matrix A_inv;                                                                      //
            Matrix B;                                                                          //
            A_inv = (I - A0/2.f).inv();                                                        //
            B     = (expf(-nudt) * I + A0/2.f);                                                //
            JJ    = (I + theta * A_inv).inv() * (A_inv * B - theta * A_inv); //M6              //
            JE    = 4.f * theta * (I + theta * A_inv).inv() * A_inv; //M7                      //
            /////////////////////////////////////////////////////////////////////////////////////
            #endif

            #if _3D
                float Jx0 = Jx0_g[ijk];
                float Jy0 = Jy0_g[ijk];
                float Jz0 = Jz0_g[ijk];

                float  Ex =  Ex_g[ijk];
                float  Ey =  Ey_g[ijk];
                float  Ez =  Ez_g[ijk];

                //int im1jp1k =  idx - 1  +     x_width * (idy + 1) + (x_width * y_width) *  idz;
                //int ip1jm1k = (idx + 1) +     x_width * (idy - 1) + (x_width * y_width) *  idz;

                //int ijk     =  idx      +     x_width *  idy      + (x_width * y_width) *  idz;
                //int ijkp1   =  idx      +     x_width *  idy      + (x_width * y_width) * (idz + 1);

                //int im1jk   = (idx - 1) +     x_width *  idy      + (x_width * y_width) *  idz;
                //int im1jkp1 = (idx - 1) +     x_width *  idy      + (x_width * y_width) * (idz + 1);

                //int ijm1k   =  idx      +     x_width * (idy - 1) + (x_width * y_width) *  idz;
                //int ijm1kp1 =  idx      +     x_width * (idy - 1) + (x_width * y_width) * (idz + 1);

                //int ijk     =  idx      +     x_width *  idy      + (x_width * y_width) *  idz;
                //int ijkm1   =  idx      +     x_width *  idy      + (x_width * y_width) * (idz - 1);

                //int ip1jk   = (idx + 1) +     x_width *  idy      + (x_width * y_width) *  idz;
                //int ip1jkm1 = (idx + 1) +     x_width *  idy      + (x_width * y_width) * (idz - 1);

                //int ijp1k   =  idx      +     x_width * (idy + 1) + (x_width * y_width) *  idz;
                //int ijp1km1 =  idx      +     x_width * (idy + 1) + (x_width * y_width) * (idz - 1);

                #if _J_intrpolate == 1
                int im1jk   = (idx - 1) +     x_width *  idy      + (x_width * y_width) *  idz;
                int ijm1k   =  idx      +     x_width * (idy - 1) + (x_width * y_width) *  idz;
                int ijkm1   =  idx      +     x_width *  idy      + (x_width * y_width) * (idz - 1);

                    if((idy > 0) && (idz > 0)){
                    int ip1jm1k = (idx + 1) +     x_width * (idy - 1) + (x_width * y_width) *  idz;
                    int ip1jk   = (idx + 1) +     x_width *  idy      + (x_width * y_width) *  idz;
                    int ip1jkm1 = (idx + 1) +     x_width *  idy      + (x_width * y_width) * (idz - 1);
                    Jx_g[ijk] = JJ.e11 *          Jx0
                              + JJ.e12 * 0.25f * (Jy0 + Jy0_g[ijm1k] + Jy0_g[ip1jk]   + Jy0_g[ip1jm1k])
                              + JJ.e13 * 0.25f * (Jz0 + Jz0_g[ip1jk] + Jz0_g[ip1jkm1] + Jz0_g[ijkm1])
                              + JE.e11 *          Ex                                                    
                              + JE.e12 * 0.25f * (Ey  + Ey_g[ijm1k]  + Ey_g[ip1jk]    + Ey_g[ip1jm1k]) 
                              + JE.e13 * 0.25f * (Ez  + Ez_g[ip1jk]  + Ez_g[ip1jkm1]  + Ez_g[ijkm1]);
                    } 

                    if((idx > 0) && (idz > 0)){
                    int im1jp1k =  idx - 1  +     x_width * (idy + 1) + (x_width * y_width) *  idz;
                    int ijp1k   =  idx      +     x_width * (idy + 1) + (x_width * y_width) *  idz;
                    int ijp1km1 =  idx      +     x_width * (idy + 1) + (x_width * y_width) * (idz - 1);
                    Jy_g[ijk] = JJ.e21 * 0.25f * (Jx0 + Jx0_g[im1jk] + Jx0_g[ijp1k]   + Jx0_g[im1jp1k])
                              + JJ.e22 *          Jy0                                 
                              + JJ.e23 * 0.25f * (Jz0 + Jz0_g[ijp1k] + Jz0_g[ijkm1]   + Jz0_g[ijp1km1])
                              + JE.e21 * 0.25f * (Ex  + Ex_g[im1jk]  + Ex_g[ijp1k]    + Ex_g[im1jp1k])
                              + JE.e22 *          Ey                                                     
                              + JE.e23 * 0.25f * (Ez  + Ez_g[ijp1k]  + Ez_g[ijkm1]    + Ez_g[ijp1km1]);                           
                    } 

                    if((idx > 0) && (idy > 0)){
                    int ijkp1   =  idx      +     x_width *  idy      + (x_width * y_width) * (idz + 1);
                    int im1jkp1 = (idx - 1) +     x_width *  idy      + (x_width * y_width) * (idz + 1);
                    int ijm1kp1 =  idx      +     x_width * (idy - 1) + (x_width * y_width) * (idz + 1);
                    Jz_g[ijk] = JJ.e31 * 0.25f * (Jx0 + Jx0_g[im1jk] + Jx0_g[ijkp1]   + Jx0_g[im1jkp1]) 
                              + JJ.e32 * 0.25f * (Jy0 + Jy0_g[ijm1k] + Jy0_g[ijkp1]   + Jy0_g[ijm1kp1]) 
                              + JJ.e33 *          Jz0                                 
                              + JE.e31 * 0.25f * (Ex  + Ex_g[im1jk]  + Ex_g[ijkp1]    + Ex_g[im1jkp1])
                              + JE.e32 * 0.25f * (Ey  + Ey_g[ijm1k]  + Ey_g[ijkp1]    + Ey_g[ijm1kp1])
                              + JE.e33 *          Ez;
                    }
                #elif _J_intrpolate == 2
                    int im1jk   = (idx - 1) +     x_width *  idy      + (x_width * y_width) *  idz;
                    int ijm1k   =  idx      +     x_width * (idy - 1) + (x_width * y_width) *  idz;
                    int ijkm1   =  idx      +     x_width *  idy      + (x_width * y_width) * (idz - 1);
                    if((idx > 0) && (idy > 0)){
                    Ex = (Ex_g[ijk] + Ex_g[im1jk])/2.f;
                    Ey = (Ey_g[ijk] + Ey_g[ijm1k])/2.f;
                    Ez = (Ez_g[ijk] + Ez_g[ijkm1])/2.f;
                                               Jx_g[ijk] = JJ.e11 * Jx0 + JJ.e12 * Jy0 + JJ.e13 * Jz0
                                                         + JE.e11 * Ex  + JE.e12 * Ey  + JE.e13 * Ez;
                                               Jy_g[ijk] = JJ.e21 * Jx0 + JJ.e22 * Jy0 + JJ.e23 * Jz0
                                                         + JE.e21 * Ex  + JE.e22 * Ey  + JE.e23 * Ez;                     
                                               Jz_g[ijk] = JJ.e31 * Jx0 + JJ.e32 * Jy0 + JJ.e33 * Jz0
                                                         + JE.e31 * Ex  + JE.e32 * Ey  + JE.e33 * Ez;
                    }
                #else
                    if((idy > 0) && (idz > 0)) Jx_g[ijk] = JJ.e11 * Jx0 + JJ.e12 * Jy0 + JJ.e13 * Jz0
                                                         + JE.e11 * Ex  + JE.e12 * Ey  + JE.e13 * Ez; 
                    if((idx > 0) && (idz > 0)) Jy_g[ijk] = JJ.e21 * Jx0 + JJ.e22 * Jy0 + JJ.e23 * Jz0
                                                         + JE.e21 * Ex  + JE.e22 * Ey  + JE.e23 * Ez; 
                    if((idx > 0) && (idy > 0)) Jz_g[ijk] = JJ.e31 * Jx0 + JJ.e32 * Jy0 + JJ.e33 * Jz0
                                                         + JE.e31 * Ex  + JE.e32 * Ey  + JE.e33 * Ez; 
                    
                    //if((idx < _nPML ) || (idx > x_width - 1 - _nPML) || (idy < _nPML) || (idy > y_width - 1 - _nPML)){
                    //if((idy > 0) && (idz > 0)){
                    //int ip1jm1k = (idx + 1) +     x_width * (idy - 1) + (x_width * y_width) *  idz;
                    //int ip1jk   = (idx + 1) +     x_width *  idy      + (x_width * y_width) *  idz;
                    //int ip1jkm1 = (idx + 1) +     x_width *  idy      + (x_width * y_width) * (idz - 1);
                    //Jx_g[ijk] = JJ.e11 *          Jx0
                    //          + JJ.e12 * 0.25f * (Jy0 + Jy0_g[ijm1k] + Jy0_g[ip1jk]   + Jy0_g[ip1jm1k])
                    //          + JJ.e13 * 0.25f * (Jz0 + Jz0_g[ip1jk] + Jz0_g[ip1jkm1] + Jz0_g[ijkm1])
                    //          + JE.e11 *          Ex                                                    
                    //          + JE.e12 * 0.25f * (Ey  + Ey_g[ijm1k]  + Ey_g[ip1jk]    + Ey_g[ip1jm1k]) 
                    //          + JE.e13 * 0.25f * (Ez  + Ez_g[ip1jk]  + Ez_g[ip1jkm1]  + Ez_g[ijkm1]);
                    //} 
                    //if((idx > 0) && (idz > 0)){
                    //int im1jp1k =  idx - 1  +     x_width * (idy + 1) + (x_width * y_width) *  idz;
                    //int ijp1k   =  idx      +     x_width * (idy + 1) + (x_width * y_width) *  idz;
                    //int ijp1km1 =  idx      +     x_width * (idy + 1) + (x_width * y_width) * (idz - 1);
                    //Jy_g[ijk] = JJ.e21 * 0.25f * (Jx0 + Jx0_g[im1jk] + Jx0_g[ijp1k]   + Jx0_g[im1jp1k])
                    //          + JJ.e22 *          Jy0                                 
                    //          + JJ.e23 * 0.25f * (Jz0 + Jz0_g[ijp1k] + Jz0_g[ijkm1]   + Jz0_g[ijp1km1])
                    //          + JE.e21 * 0.25f * (Ex  + Ex_g[im1jk]  + Ex_g[ijp1k]    + Ex_g[im1jp1k])
                    //          + JE.e22 *          Ey                                                     
                    //          + JE.e23 * 0.25f * (Ez  + Ez_g[ijp1k]  + Ez_g[ijkm1]    + Ez_g[ijp1km1]);                           
                    //} 
                    //if((idx > 0) && (idy > 0)){
                    //int ijkp1   =  idx      +     x_width *  idy      + (x_width * y_width) * (idz + 1);
                    //int im1jkp1 = (idx - 1) +     x_width *  idy      + (x_width * y_width) * (idz + 1);
                    //int ijm1kp1 =  idx      +     x_width * (idy - 1) + (x_width * y_width) * (idz + 1);
                    //Jz_g[ijk] = JJ.e31 * 0.25f * (Jx0 + Jx0_g[im1jk] + Jx0_g[ijkp1]   + Jx0_g[im1jkp1]) 
                    //          + JJ.e32 * 0.25f * (Jy0 + Jy0_g[ijm1k] + Jy0_g[ijkp1]   + Jy0_g[ijm1kp1]) 
                    //          + JJ.e33 *          Jz0                                 
                    //          + JE.e31 * 0.25f * (Ex  + Ex_g[im1jk]  + Ex_g[ijkp1]    + Ex_g[im1jkp1])
                    //          + JE.e32 * 0.25f * (Ey  + Ey_g[ijm1k]  + Ey_g[ijkp1]    + Ey_g[ijm1kp1])
                    //          + JE.e33 *          Ez;
                    //}
                    //}
                    
                #endif
            #else
            //if((idx > 0) && (idy > 0)){
                float Jx0 = Jx0_g[ijk];
                float Jy0 = Jy0_g[ijk];
                float  Ex =  Ex_g[ijk];
                float  Ey =  Ey_g[ijk];
                #if _Full2D
                float  Ez =  Ez_g[ijk];
                float Jz0 = Jz0_g[ijk];               

                    #if _J_intrpolate == 1
                        int ijm1k = idx        +     x_width * (idy - 1) ;               
                        int im1jk = (idx - 1)  +     x_width * idy;       
                        int ijp1k   = idx      +     x_width * (idy + 1) ;
                        int ip1jk   = (idx + 1)+     x_width * idy;       
                        int im1jp1k = idx - 1  +     x_width * (idy + 1) ;
                        int ip1jm1k = (idx + 1)+     x_width * (idy - 1) ;
                        if(idy > 0){
                        Jx_g[ijk] = JJ.e11 *           Jx0
                                  + JJ.e12 * 0.25f *  (Jy0 + Jy0_g[ijm1k] + Jy0_g[ip1jk] + Jy0_g[ip1jm1k])
                                  + JJ.e13 * 0.5f  *  (Jz0 + Jz0_g[ip1jk])
                                  + JE.e11 *            Ex                                                    
                                  + JE.e12 * 0.25f *   (Ey + Ey_g[ijm1k] + Ey_g[ip1jk] + Ey_g[ip1jm1k]) 
                                  + JE.e13 * 0.5f  *   (Ez + Ez_g[ip1jk]);}
                        if(idx > 0){
                        Jy_g[ijk] = JJ.e21 * 0.25f *  (Jx0 + Jx0_g[im1jk] + Jx0_g[ijp1k] + Jx0_g[im1jp1k])
                                  + JJ.e22 *           Jy0
                                  + JJ.e23 * 0.5f  *  (Jz0 + Jz0_g[ijp1k])
                                  + JE.e21 * 0.25f *   (Ex + Ex_g[im1jk] + Ex_g[ijp1k] + Ex_g[im1jp1k])
                                  + JE.e22 *            Ey                                                   
                                  + JE.e23 * 0.5f  *   (Ez + Ez_g[ijp1k]);}
                        if((idx > 0) && (idy > 0)){
                        Jz_g[ijk] = JJ.e31 * 0.5f  *  (Jx0 + Jx0_g[im1jk]) 
                                  + JJ.e32 * 0.5f  *  (Jy0 + Jy0_g[ijm1k]) 
                                  + JJ.e33 *           Jz0 
                                  + JE.e31 * 0.5f  *   (Ex + Ex_g[im1jk])
                                  + JE.e32 * 0.5f  *   (Ey + Ey_g[ijm1k])
                                  + JE.e33 *            Ez;}
                    #elif _J_intrpolate == 2
                        int im1jk = (idx - 1)  +     x_width * idy;
                        int ijm1k = idx        +     x_width * (idy - 1) ;  
                        if((idx > 0) && (idy > 0)){                             
                            Ex = (Ex_g[ijk] + Ex_g[im1jk])/2.f;
                            Ey = (Ey_g[ijk] + Ey_g[ijm1k])/2.f;
                                                       Jx_g[ijk] = JJ.e11 * Jx0 + JJ.e12 * Jy0 + JJ.e13 * Jz0
                                                                 + JE.e11 * Ex  + JE.e12 * Ey  + JE.e13 * Ez;
                                                       Jy_g[ijk] = JJ.e21 * Jx0 + JJ.e22 * Jy0 + JJ.e23 * Jz0
                                                                 + JE.e21 * Ex  + JE.e22 * Ey  + JE.e23 * Ez;                     
                                                       Jz_g[ijk] = JJ.e31 * Jx0 + JJ.e32 * Jy0 + JJ.e33 * Jz0
                                                                 + JE.e31 * Ex  + JE.e32 * Ey  + JE.e33 * Ez;
                        }
                    #else
                    
                        if(idy > 0)                Jx_g[ijk] = JJ.e11 * Jx0 + JJ.e12 * Jy0 + JJ.e13 * Jz0
                                                             + JE.e11 * Ex  + JE.e12 * Ey  + JE.e13 * Ez; 
                        if(idx > 0)                Jy_g[ijk] = JJ.e21 * Jx0 + JJ.e22 * Jy0 + JJ.e23 * Jz0
                                                             + JE.e21 * Ex  + JE.e22 * Ey  + JE.e23 * Ez; 
                        if((idx > 0) && (idy > 0)) Jz_g[ijk] = JJ.e31 * Jx0 + JJ.e32 * Jy0 + JJ.e33 * Jz0
                                                             + JE.e31 * Ex  + JE.e32 * Ey  + JE.e33 * Ez; 

                    #endif
                    #if _Jdt == 3 
                        float l = __int2float_rn(_nPML);
                        if (idx < _nPML) l = __int2float_rn(idx);
                        if (idx > x_width - 1 - _nPML) l = __int2float_rn(-idx + x_width - 1);
                        float kappa_x = CPML_kappa(l);

                        l = __int2float_rn(_nPML);
                        if (idy < _nPML) l = __int2float_rn(idy);
                        if (idy > y_width - 1 - _nPML)  l = __int2float_rn(-idy + y_width - 1);
                        float kappa_y = CPML_kappa(l);

                        float dHx = 0.f;
                        float dHy = 0.f;
                        float dHz = 0.f;
                        #if _J_intrpolate == 1
                            if(idy > 0) dHx = _Courant * ( (Hz_g[ijk] - Hz_g[ijm1k])/kappa_y);
                            if(idx > 0) dHy = _Courant * (-(Hz_g[ijk] - Hz_g[im1jk])/kappa_x);
                        #elif _J_intrpolate == 2
                            if((idx > 0) && (idy > 0)){
                                int im1jm1k= idx-1 + x_width * (idy-1);
                                dHx = _Courant * ( (Hz_g[ijk] - Hz_g[ijm1k])/kappa_y + (Hz_g[im1jk] - Hz_g[im1jm1k])/kappa_y)/2.f;
                                dHy = _Courant * (-(Hz_g[ijk] - Hz_g[im1jk])/kappa_x - (Hz_g[ijm1k] - Hz_g[im1jm1k])/kappa_x)/2.f;
                            }
                        #else
                            int im1jk= idx-1 + x_width * idy;
                            int ijm1k= idx   + x_width *(idy-1);
                            if(idy > 0) dHx = _Courant * ( (Hz_g[ijk] - Hz_g[ijm1k])/kappa_y);
                            if(idx > 0) dHy = _Courant * (-(Hz_g[ijk] - Hz_g[im1jk])/kappa_x);
                        #endif                    
                        if((idx > 0) && (idy > 0)) dHz = _Courant * ((Hy_g[ijk] - Hy_g[im1jk])/kappa_x - (Hx_g[ijk] - Hx_g[ijm1k])/kappa_y);
                        if(idy > 0)                Jx_g[ijk] += (JE.e11 * dHx  + JE.e12 * dHy  + JE.e13 * dHz)/2.f;
                        if(idx > 0)                Jy_g[ijk] += (JE.e21 * dHx  + JE.e22 * dHy  + JE.e23 * dHz)/2.f;
                        if((idx > 0) && (idy > 0)) Jz_g[ijk] += (JE.e31 * dHx  + JE.e32 * dHy  + JE.e33 * dHz)/2.f;
                    #endif
         
                #else
                //if((idy > 0)){
                //Jx_g[ijk] = JJ.e11 *           Jx0
                //          + JJ.e12 * 0.25f *  (Jy0 + Jy0_g[ijm1k] + Jy0_g[ip1jk] + Jy0_g[ip1jm1k])
                //          + JE.e11 *            Ex                                                    
                //          + JE.e12 * 0.25f *   (Ey + Ey_g[ijm1k] + Ey_g[ip1jk] + Ey_g[ip1jm1k]);
                //} 
                //if((idx > 0)){
                //Jy_g[ijk] = JJ.e21 * 0.25f *  (Jx0 + Jx0_g[im1jk] + Jx0_g[ijp1k] + Jx0_g[im1jp1k])
                //          + JJ.e22 *           Jy0
                //          + JE.e21 * 0.25f *   (Ex + Ex_g[im1jk] + Ex_g[ijp1k] + Ex_g[im1jp1k])
                //          + JE.e22 *            Ey;
                //} 
                if((idx > 0) && (idy > 0)){

                //float  Hz =  Hz_g[ijk];
                //float dHx = _Courant * ( (Hz - Hz_g[ijm1k]));                                                                 
                //float dHy = _Courant * (-(Hz - Hz_g[im1jk]));                            
                float Jy0_x= Jy0;
                float Ey_x = Ey; 
                float Jx0_y= Jx0;
                float Ex_y = Ex;                
                //if((idx < _nPML) || (idx > x_width - 1 - _nPML) || (idy < _nPML) || (idy > y_width - 1 - _nPML)){
                //    Jy0_x =  0.25f *  (Jy0 + Jy0_g[ijm1k] + Jy0_g[ip1jk] + Jy0_g[ip1jm1k]);
                //    Ey_x  =  0.25f *   (Ey + Ey_g[ijm1k] + Ey_g[ip1jk] + Ey_g[ip1jm1k]);

                //    Jx0_y =  0.25f *  (Jx0 + Jx0_g[im1jk] + Jx0_g[ijp1k] + Jx0_g[im1jp1k]);
                //    Ex_y  =  0.25f *   (Ex + Ex_g[im1jk] + Ex_g[ijp1k] + Ex_g[im1jp1k]);
                //}

                Jx_g[ijk] = JJ.e11 *     Jx0
                          + JJ.e12 *     Jy0_x

                          + JE.e11 *     Ex                                               
                          + JE.e12 *     Ey_x;

                          //+ JE.e11/2 *    dHx                                                   
                          //+ JE.e12/2 *    dHy;

                Jy_g[ijk] = JJ.e21 *     Jx0_y
                          + JJ.e22 *     Jy0

                          + JE.e21 *     Ex_y
                          + JE.e22 *     Ey;

                          //+ JE.e21/2 *    dHx                                                   
                          //+ JE.e22/2 *    dHy;

                } 

                #endif
            //}
            #endif
        }
        #if _J_intrpolate == 2
        if((idx == 0) || (idx == x_width - 1) || (idy == 0) || (idy == y_width -1)) {
        int ijk = idx +     x_width * idy;
        #if _3D
            ijk += (x_width * y_width) *  idz;
        #endif
        	Jx_g[ijk]=0.f;
        	Jy_g[ijk]=0.f;
        	Jz_g[ijk]=0.f;
        }
        #endif
    }

}
__global__ void update_3D_E_CPML(
                                  float * __restrict__ Ex_g, 
                                  float * __restrict__ Ey_g, 
                                  float * __restrict__ Ez_g, 
                            const float * __restrict__ Hx_g, 
                            const float * __restrict__ Hy_g, 
                            const float * __restrict__ Hz_g, 
                            const float * __restrict__ Jx_g, 
                            const float * __restrict__ Jy_g, 
                            const float * __restrict__ Jz_g, 
                            const float * __restrict__ J0x_g, 
                            const float * __restrict__ J0y_g, 
                            const float * __restrict__ J0z_g, 
                                  float * __restrict__ pEyx_g, 
                                  float * __restrict__ pEzx_g, 
                                  float * __restrict__ pExy_g, 
                                  float * __restrict__ pEzy_g)
{
    int   idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int   idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    //int   x_width = blockDim.x * gridDim.x;
    //int   y_width = blockDim.y * gridDim.y;

    #if _3D
    for(int idz = 1; idz < z_width - 1; idz++) // loop over z coordinate (numpy axis 0)
    #endif
    {
        #if _3D
            int ijk = idx   + x_width * idy   + (x_width * y_width) *  idz;
            int i1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ij1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijk1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);
        #else
            int ijk = idx   + x_width * idy ;
            int i1jk= idx-1 + x_width * idy ;
            int ij1k= idx   + x_width *(idy-1);
        #endif

        //if((idx < x_width - 1) && (idy < y_width - 1)){
            #if _Full2D
            float Hx = Hx_g[ijk];
            float Hy = Hy_g[ijk];
            float pEzx    =  0.f;
            float pEzy    =  0.f;
            #endif
            float Hz = Hz_g[ijk];

            float pEyx    =  0.f; 
            float pExy    =  0.f;
            
            float kappa_x = 1.f;
            float kappa_y = 1.f;
            if ((idx < _nPML) || (idx > x_width - 1 - _nPML)){
                float l = __int2float_rn(idx);
            #if _3D
                int ijk_x1PML = idx +     _nPML * idy  + (2 * _nPML * y_width) *  idz;
            #else
                int ijk_x1PML = idx +     _nPML * idy;
            #endif
                if (idx > x_width - 1 - _nPML){
                    l = __int2float_rn(-idx + x_width - 1);
                    #if _3D
                        ijk_x1PML = (idx - 1 - (x_width - 1 - _nPML)) + idy * _nPML + _nPML * y_width + (2 * _nPML * y_width) *  idz;
                    #else
                        ijk_x1PML = (idx - 1 - (x_width - 1 - _nPML)) + idy * _nPML + _nPML * y_width;
                    #endif
                }
                kappa_a_b k_a_b_x = CPML_grading(l);

                if((idx > 0))
                    pEyx  = k_a_b_x.b * pEyx_g[ijk_x1PML]  + k_a_b_x.a * _Courant * ((Hz - Hz_g[i1jk]));
                pEyx_g[ijk_x1PML] = pEyx;
                #if _Full2D
                if((idx > 0) && (idy > 0))
                    pEzx  = k_a_b_x.b * pEzx_g[ijk_x1PML]  + k_a_b_x.a * _Courant * ((Hy - Hy_g[i1jk]));
                pEzx_g[ijk_x1PML] = pEzx;
                #endif

                kappa_x=k_a_b_x.kappa;
            }
            if((idy < _nPML) || (idy > y_width - 1 - _nPML)){
                float l = float(idy);
                #if _3D
                    int ijk_y1PML = idx +     x_width * idy + (2 * _nPML * x_width) *  idz;
                #else
                    int ijk_y1PML = idx +     x_width * idy;
                #endif
                if (idy > y_width - 1 - _nPML) {
		            l = __int2float_rn(-idy + y_width - 1);
                    #if _3D
                        ijk_y1PML = idx + (idy - 1 - (y_width - 1 - _nPML)) * x_width +  x_width * _nPML + (2 * _nPML * x_width) *  idz;
                    #else
                        ijk_y1PML = idx + (idy - 1 - (y_width - 1 - _nPML)) * x_width +  x_width * _nPML;
                    #endif
	            }
                kappa_a_b k_a_b_y = CPML_grading(l);

                if(idy > 0)
                    pExy  = k_a_b_y.b * pExy_g[ijk_y1PML] + k_a_b_y.a * _Courant * ((Hz - Hz_g[ij1k]));
                pExy_g[ijk_y1PML]= pExy;

                #if _Full2D
                if((idx > 0) && (idy > 0))
                    pEzy  = k_a_b_y.b * pEzy_g[ijk_y1PML] + k_a_b_y.a * _Courant * ((Hx - Hx_g[ij1k]));
                pEzy_g[ijk_y1PML]= pEzy;
                #endif

                kappa_y=k_a_b_y.kappa;
            }

            #if _3D
                // Ex,Jx; Ey,Jy; Ez,Jz co-located
                if((idy > 0) && (idz > 0)){
                    Ex_g[ijk] += _Courant * ( (Hz - Hz_g[ij1k])/kappa_y - (Hy - Hy_g[ijk1])        ) + pExy;
                    #if _Jdt == 3
                        #if _J_intrpolate == 2
                            int ip1jk= idx+1 + x_width * idy   + (x_width * y_width) *  idz;
                            Ex_g[ijk] -= (Jx_g[ijk]+J0x_g[ijk]+Jx_g[ip1jk]+J0x_g[ip1jk])/4.f;
                        #else
                            Ex_g[ijk] -= (Jx_g[ijk]+J0x_g[ijk])/2.f;
                        #endif
                    #else
                        #if _J_intrpolate == 2
                            int ip1jk= idx+1 + x_width * idy   + (x_width * y_width) *  idz;
                            Ex_g[ijk] -= (Jx_g[ijk]+Jx_g[ip1jk])/2.f;
                        #else
                            Ex_g[ijk] -= Jx_g[ijk];
                        #endif
                    #endif
                }
                if((idx > 0) && (idz > 0)){
                    Ey_g[ijk] += _Courant * ( (Hx - Hx_g[ijk1])         - (Hz - Hz_g[i1jk])/kappa_x) - pEyx;
                    #if _Jdt == 3
                        #if _J_intrpolate == 2
                            int ijp1k= idx + x_width * (idy+1) + (x_width * y_width) *  idz ;
                            Ey_g[ijk] -= (Jy_g[ijk]+J0y_g[ijk]+Jy_g[ijp1k]+J0y_g[ijp1k])/4.f;
                        #else
                            Ey_g[ijk] -= (Jy_g[ijk]+J0y_g[ijk])/2.f;
                        #endif
                    #else
                        #if _J_intrpolate == 2
                            int ijp1k= idx + x_width * (idy+1) + (x_width * y_width) *  idz ;
                            Ey_g[ijk] -= (Jy_g[ijk]+Jy_g[ijp1k])/2.f;
                        #else
                            Ey_g[ijk] -= Jy_g[ijk];
                        #endif
                    #endif   
                }
                if((idx > 0) && (idy > 0)){
                    Ez_g[ijk] += _Courant * ( (Hy - Hy_g[i1jk])/kappa_x - (Hx - Hx_g[ij1k])/kappa_y) + pEzx - pEzy;
                    #if _Jdt == 3
                        #if _J_intrpolate == 2
                            int ijkp1= idx + x_width * idy + (x_width * y_width) *  (idz+1) ;
                            Ez_g[ijk] -= (Jz_g[ijk]+J0z_g[ijk]+Jz_g[ijkp1]+J0z_g[ijkp1])/4.f;
                        #else
                            Ez_g[ijk] -= (Jz_g[ijk]+J0z_g[ijk])/2.f; 
                        #endif
                    #else
                        #if _J_intrpolate == 2
                            int ijkp1= idx + x_width * idy + (x_width * y_width) *  (idz+1) ;
                            Ez_g[ijk] -= (Jz_g[ijk]+Jz_g[ijkp1])/2.f;
                        #else
                            Ez_g[ijk] -= Jz_g[ijk];
                        #endif
                    #endif
                }
            #else
                // Ex,Jx; Ey,Jy; Ez,Jz co-located
                if((idy > 0)){
                    Ex_g[ijk] += _Courant * ( (Hz - Hz_g[ij1k])/kappa_y)                             + pExy;
                    #if _Jdt == 3
                        #if _J_intrpolate == 2
                            int ip1jk= idx+1 + x_width * idy ;
                            Ex_g[ijk] -= (Jx_g[ijk]+J0x_g[ijk]+Jx_g[ip1jk]+J0x_g[ip1jk])/4.f;
                        #else
                            Ex_g[ijk] -= (Jx_g[ijk]+J0x_g[ijk])/2.f;
                        #endif
                    #else
                        #if _J_intrpolate == 2
                            int ip1jk= idx+1 + x_width * idy ;
                            Ex_g[ijk] -= (Jx_g[ijk]+Jx_g[ip1jk])/2.f;
                        #else
                            Ex_g[ijk] -= Jx_g[ijk];
                        #endif
                    #endif
                }                                                                                                             
                if((idx > 0)){                                                                                                
                    Ey_g[ijk] += _Courant * (-(Hz - Hz_g[i1jk])/kappa_x)                             - pEyx;
                    #if _Jdt == 3
                        #if _J_intrpolate == 2
                            int ijp1k= idx + x_width * (idy+1) ;
                            Ey_g[ijk] -= (Jy_g[ijk]+J0y_g[ijk]+Jy_g[ijp1k]+J0y_g[ijp1k])/4.f;
                        #else
                            Ey_g[ijk] -= (Jy_g[ijk]+J0y_g[ijk])/2.f;
                        #endif
                    #else
                        #if _J_intrpolate == 2
                            int ijp1k= idx + x_width * (idy+1) ;
                            Ey_g[ijk] -= (Jy_g[ijk]+Jy_g[ijp1k])/2.f;
                        #else
                            Ey_g[ijk] -= Jy_g[ijk];
                        #endif
                    #endif                   
                }                                                                                                             
                #if _Full2D                                                                                                   
                if((idx > 0) && (idy > 0)){                                                                                   
                    Ez_g[ijk] += _Courant * ( (Hy - Hy_g[i1jk])/kappa_x - (Hx - Hx_g[ij1k])/kappa_y) + pEzx - pEzy;
                    #if _Jdt == 3
                        Ez_g[ijk] -= (Jz_g[ijk]+J0z_g[ijk])/2.f;
                    #else
                        Ez_g[ijk] -= Jz_g[ijk];
                    #endif                       
                }
                #endif
            #endif

        //}
        
        if( 
        #if _PECx0
            (idx == _nPML)
        #else
            (idx == 0)
        #endif
            ||
        #if _PECxN
            (idx == x_width - 1 - _nPML)
        #else
            (idx == x_width - 1)
        #endif
        ){
        #if _3D
            int ijk = idx   + x_width * idy   + (x_width * y_width) *  idz;
        #else
            int ijk = idx   + x_width * idy ;
        #endif
        	Ey_g[ijk]=0.f;
        	Ez_g[ijk]=0.f;
        }
        
        //if((idy == 0) || (idy == y_width -1)) {
        if( 
        #if _PECy0
            (idy == _nPML)
        #else
            (idy == 0)
        #endif
            ||
        #if _PECyN
            (idy == y_width - 1 - _nPML)
        #else
            (idy == y_width - 1)
        #endif
        ){
        #if _3D
            int ijk = idx   + x_width * idy   + (x_width * y_width) *  idz;
        #else
            int ijk = idx   + x_width * idy ;
        #endif
        	Ex_g[ijk]=0.f;
         	Ez_g[ijk]=0.f;
        }
    }

}

__global__ void E_restart(int t,
                          int loc,
                        float * __restrict__ Ex_g, 
                        float * __restrict__ Ey_g, 
                        float * __restrict__ Ez_g,
                 const  float * __restrict__ Ex_restart_g,
                 const  float * __restrict__ Ey_restart_g,
                 const  float * __restrict__ Ez_restart_g)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    //int x_width = blockDim.x * gridDim.x;
    //int y_width = blockDim.y * gridDim.y;
    #if _3D
    for(int idz = 0; idz < z_width; idz++) // loop over z coordinate (numpy axis 0)
    #endif
    {
        #if _3D
            //int y_width = blockDim.y * gridDim.y;
            int ijk = idx +     x_width * idy + (x_width * y_width) *  idz;
            int ijk_s = idx + x_width * idz + x_width * z_width * t;
        #else
            int ijk = idx +     x_width * idy;
            int ijk_s = idx + x_width * t;
        #endif
        if(idy == loc){
            Ex_g[ijk]=Ex_restart_g[ijk_s];
            Ey_g[ijk]=Ey_restart_g[ijk_s];
            Ez_g[ijk]=Ez_restart_g[ijk_s];
        }
    }
}
__global__ void save_E_edge(                             int t,
                                                         int loc,
                                  const float * __restrict__ Ex_g, 
                                  const float * __restrict__ Ey_g, 
                                  const float * __restrict__ Ez_g, 
                                        float * __restrict__ Ex_restart_g,
                                        float * __restrict__ Ey_restart_g,
                                        float * __restrict__ Ez_restart_g)
                                        
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    //int x_width = blockDim.x * gridDim.x;
    //int y_width = blockDim.y * gridDim.y;

    #if _3D
    for(int idz = 0; idz < z_width; idz++) // loop over z coordinate (numpy axis 0)
    #endif
    {
        #if _3D
            //int y_width = blockDim.y * gridDim.y;
            int ijk = idx +     x_width * idy + (x_width * y_width) *  idz;
            int ijk_s = idx + x_width * idz + x_width * z_width * t;
        #else
            int ijk = idx +     x_width * idy;
            int ijk_s = idx + x_width * t;
        #endif
        if(idy == loc){
        float Ex  =  Ex_g[ijk]; float Ey  =  Ey_g[ijk]; float Ez  =  Ez_g[ijk];
            Ex_restart_g[ijk_s] = Ex;
            Ey_restart_g[ijk_s] = Ey;
            Ez_restart_g[ijk_s] = Ez;
        }
    }
}
//template<int location>
__global__ void E_source(int t,
                         int location,
                       float ramp_up,
                       float n_steps_per_lambda,
                       float z0,
                       float w_0,
                       float x_sofs,
                       float k_x,
                       float k_z,
                       float E0_x,
                       float phy0_x,
                       float E0_y,
                       float phy0_y,
                       float E0_z,
                       float phy0_z,
                       float * __restrict__ Ex_g, 
                       float * __restrict__ Ey_g, 
                       float * __restrict__ Ez_g,
                       float * __restrict__ Hx_g, 
                       float * __restrict__ Hy_g, 
                       float * __restrict__ Hz_g)                       
{
    int   idx     = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    //int   x_width = blockDim.x * gridDim.x;

    #if _3D
    int   idy_z   = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    if((idy_z > 0) && (idy_z < z_width - 2))
    #endif
    {
        //generic coordinates
        //int ijk = idx +     x_width * idy + (x_width * y_width) *  idz;
        
        #if _3D        
        //int ijk = idx + x_width * _nPML + (x_width * y_width) * idy_z;
        int ijk; int width;
        switch(location){
            default:      ijk = idx + x_width * _nPML;                width = x_width; break; //y0 
            case 2:       ijk = _nPML + x_width * idx;                width = y_width; break; //x0
            case 3:       ijk = idx + x_width * (y_width - 1 - _nPML);width = x_width; break; //yN
            case 4:       ijk = (x_width - 1 - _nPML) + x_width * idx;width = y_width; break; //xN
        }
        ijk+= (x_width * y_width) * idy_z;
        #else
        int ijk; int width;
        switch(location){
            default:      ijk = idx + x_width * _nPML;                width = x_width; break; //y0 
            case 2:       ijk = _nPML + x_width * idx;                width = y_width; break; //x0
            case 3:       ijk = idx + x_width * (y_width - 1 - _nPML);width = x_width; break; //yN
            case 4:       ijk = (x_width - 1 - _nPML) + x_width * idx;width = y_width; break; //xN
        }
        #endif
        if((idx > _nPML) && (idx < width - _nPML)){
            float tn   = __saturatef(t/(ramp_up*n_steps_per_lambda/fabsf(_Courant))); //__saturatef()
            //float tn2  = __saturatef(10.f-t/(10.f*n_steps_per_lambda/fabsf(_Courant))); //__saturatef()
            //float rup = powf(tn,2)*(3.f-2.f*tn);//*powf(tn2,2)*(3.f-2.f*tn2);
            float rup = 0.5f - 0.5f * cospif(tn);
            float X_box   = float(idx-x_sofs);
            #if _3D
                float Z_box   = float(idy_z-z_width/2);
                float z   = (X_box*k_x+Z_box*k_z)+z0;
                float r   = sqrtf(powf(X_box-k_x*(X_box*k_x+Z_box*k_z),2)
                                 +(1.f-k_x*k_x-k_z*k_z)*powf((X_box*k_x+Z_box*k_z),2)
                                 +powf(Z_box-k_z*(X_box*k_x+Z_box*k_z),2));
            #else
                float r   = fabsf(X_box*sqrtf(1.f-k_x*k_x));// + k_x * 0.5f);
                float z   = (X_box*k_x)+z0;// - 0.5f*sqrtf(1.f-k_x*k_x);
            #endif
                float z_R = CUDART_PI_F * w_0 * w_0 / n_steps_per_lambda;  
                float w   = w_0 * sqrtf(1.f+z*z/(z_R*z_R));

                float GBeam_Amplitude = rup * (w_0/w) * expf(-fabsf(powf(r,2))/powf(w,2));
                float R   = z * (1.f + z_R * z_R / (z*z));
// - atanf(z/z_R)/CUDART_PI_F
            //switch(location){
            //    case 1:
            //    case 3: Ex_g[ijk] -= _Courant * GBeam_Amplitude * E0_x * cospif(-phy0_x + 2.0f * (-_Courant * t + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda); break; //y0
            //    case 2:
            //    case 4: Ey_g[ijk] -= _Courant * GBeam_Amplitude * E0_y * cospif(-phy0_y + 2.0f * (-_Courant * t + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda); break; //x0
            //}    
           
             Ex_g[ijk] -= _Courant * GBeam_Amplitude * E0_x * cospif(-phy0_x + 2.0f * (-_Courant * t + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda);
             Ey_g[ijk] -= _Courant * GBeam_Amplitude * E0_y * cospif(-phy0_y + 2.0f * (-_Courant * t + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda);
           
           // this is implementation of TS/SF for y=0 plane. It wrorks, but there is still 1% wave going wrong direction.
           // this goes up 
           //case 3: Ex_g[ijk] -= _Courant * GBeam_Amplitude * E0_x * cospif(-phy0_x + 2.0f * (-_Courant * (t-0.5f) + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda); break; //y0
           //int ijk1 = idx + x_width * (_nPML - 1); //y0-1
           // r   = fabsf(X_box*sqrtf(1.f-k_x*k_x) + k_x * 0.5f);
           // z   = (X_box*k_x)+z0 - 0.5f*sqrtf(1.f-k_x*k_x);
           // w   = w_0 * sqrtf(1.f+z*z/(z_R*z_R));
           // GBeam_Amplitude = rup * (w_0/w) * expf(-fabsf(powf(r,2))/powf(w,2));
           // R   = z * (1.f + z_R * z_R / (z*z));
           // Hz_g[ijk1] += _Courant * GBeam_Amplitude * E0_x * sinpif(-phy0_x + 2.0f * (-_Courant * (t+0.5f) + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda);
            
                #if _Full2D
                Ez_g[ijk] -= _Courant * GBeam_Amplitude * E0_z * cospif(-phy0_z + 2.0f * (-_Courant * t + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda);
                #endif

        }
    }
}
//Sx0x,SxNx,Sx0y,SxNy,Sx0z,SxNz,Sy0x,SyNx,Sy0y,SyNy,Sy0z,SyNz,Sz0x,SzNx,Sz0y,SzNy,Sz0z,SzNz,SzIx,SzIy,SzIz
__global__ void S_Diagnostic_3D(  const float * __restrict__ Ex_g, 
                                  const float * __restrict__ Ey_g, 
                                  const float * __restrict__ Ez_g, 
                                  const float * __restrict__ Hx_g, 
                                  const float * __restrict__ Hy_g, 
                                  const float * __restrict__ Hz_g,
                                  const float * __restrict__ Jx_g,
                                  const float * __restrict__ Jy_g,
                                  const float * __restrict__ Jz_g,
                                        float * __restrict__ Sx0x_g,
                                        float * __restrict__ Sx0y_g,
                                        float * __restrict__ Sx0z_g,
                                        float * __restrict__ SxNx_g,
                                        float * __restrict__ SxNy_g,
                                        float * __restrict__ SxNz_g,
                                        float * __restrict__ Sy0x_g,
                                        float * __restrict__ Sy0y_g,
                                        float * __restrict__ Sy0z_g,
                                        float * __restrict__ SyNx_g,
                                        float * __restrict__ SyNy_g,
                                        float * __restrict__ SyNz_g,
                                        float * __restrict__ Sz0x_g,
                                        float * __restrict__ Sz0y_g,
                                        float * __restrict__ Sz0z_g,
                                        float * __restrict__ SzNx_g,
                                        float * __restrict__ SzNy_g,
                                        float * __restrict__ SzNz_g,
                                        float * __restrict__ SzIx_g,
                                        float * __restrict__ SzIy_g,
                                        float * __restrict__ SzIz_g)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    //int x_width = blockDim.x * gridDim.x;
    //int y_width = blockDim.y * gridDim.y;

    if((idy > 0) && (idx > 0)){
    for(int idz = 1; idz < z_width; idz++) // loop over z coordinate (numpy axis 0)
    {
        int ijk = idx +     x_width * idy + (x_width * y_width) *  idz;

        if(idx == (_nPML + 2)){
            int im1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ijm1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijkm1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);  
            int im1jm1k= idx-1 + x_width * (idy-1)   + (x_width * y_width) *  idz;
            int im1jkm1= idx-1 + x_width * idy       + (x_width * y_width) *  (idz -1);
            int ijm1km1= idx   + x_width * (idy-1)   + (x_width * y_width) *  (idz -1);
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  0.5f * (Ez_g[ijk] + Ez_g[ijkm1]);
            float Hx  = 0.25f * (Hx_g[ijk] + Hx_g[ijm1k] + Hx_g[ijkm1] + Hx_g[ijm1km1]);
            float Hy  = 0.25f * (Hy_g[ijk] + Hy_g[im1jk] + Hy_g[ijkm1] + Hy_g[im1jkm1]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]);        
            
            Sx0x_g[idy + y_width * idz]+= Ey * Hz - Ez * Hy;
            Sx0y_g[idy + y_width * idz]+= Ez * Hx - Ex * Hz;
            Sx0z_g[idy + y_width * idz]+= Ex * Hy - Ey * Hx;
        }
        if(idx == (x_width - 3 - _nPML)){
            int im1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ijm1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijkm1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);  
            int im1jm1k= idx-1 + x_width * (idy-1)   + (x_width * y_width) *  idz;
            int im1jkm1= idx-1 + x_width * idy       + (x_width * y_width) *  (idz -1);
            int ijm1km1= idx   + x_width * (idy-1)   + (x_width * y_width) *  (idz -1);
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  0.5f * (Ez_g[ijk] + Ez_g[ijkm1]);
            float Hx  = 0.25f * (Hx_g[ijk] + Hx_g[ijm1k] + Hx_g[ijkm1] + Hx_g[ijm1km1]);
            float Hy  = 0.25f * (Hy_g[ijk] + Hy_g[im1jk] + Hy_g[ijkm1] + Hy_g[im1jkm1]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]); 

            SxNx_g[idy + y_width * idz]+= Ey * Hz - Ez * Hy;
            SxNy_g[idy + y_width * idz]+= Ez * Hx - Ex * Hz;
            SxNz_g[idy + y_width * idz]+= Ex * Hy - Ey * Hx;
        }
        if(idy == (_nPML + 2)){
            int im1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ijm1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijkm1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);  
            int im1jm1k= idx-1 + x_width * (idy-1)   + (x_width * y_width) *  idz;
            int im1jkm1= idx-1 + x_width * idy       + (x_width * y_width) *  (idz -1);
            int ijm1km1= idx   + x_width * (idy-1)   + (x_width * y_width) *  (idz -1);
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  0.5f * (Ez_g[ijk] + Ez_g[ijkm1]);
            float Hx  = 0.25f * (Hx_g[ijk] + Hx_g[ijm1k] + Hx_g[ijkm1] + Hx_g[ijm1km1]);
            float Hy  = 0.25f * (Hy_g[ijk] + Hy_g[im1jk] + Hy_g[ijkm1] + Hy_g[im1jkm1]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]); 

            Sy0x_g[idx + x_width * idz]+= Ey * Hz - Ez * Hy;
            Sy0y_g[idx + x_width * idz]+= Ez * Hx - Ex * Hz;
            Sy0z_g[idx + x_width * idz]+= Ex * Hy - Ey * Hx;
        }
        if(idy == (y_width - 3 - _nPML)){
            int im1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ijm1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijkm1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);  
            int im1jm1k= idx-1 + x_width * (idy-1)   + (x_width * y_width) *  idz;
            int im1jkm1= idx-1 + x_width * idy       + (x_width * y_width) *  (idz -1);
            int ijm1km1= idx   + x_width * (idy-1)   + (x_width * y_width) *  (idz -1);
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  0.5f * (Ez_g[ijk] + Ez_g[ijkm1]);
            float Hx  = 0.25f * (Hx_g[ijk] + Hx_g[ijm1k] + Hx_g[ijkm1] + Hx_g[ijm1km1]);
            float Hy  = 0.25f * (Hy_g[ijk] + Hy_g[im1jk] + Hy_g[ijkm1] + Hy_g[im1jkm1]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]); 

            SyNx_g[idx + x_width * idz]+= Ey * Hz - Ez * Hy;
            SyNy_g[idx + x_width * idz]+= Ez * Hx - Ex * Hz;
            SyNz_g[idx + x_width * idz]+= Ex * Hy - Ey * Hx;
        }
        if(idz == 1){
            int im1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ijm1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijkm1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);  
            int im1jm1k= idx-1 + x_width * (idy-1)   + (x_width * y_width) *  idz;
            int im1jkm1= idx-1 + x_width * idy       + (x_width * y_width) *  (idz -1);
            int ijm1km1= idx   + x_width * (idy-1)   + (x_width * y_width) *  (idz -1);
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  0.5f * (Ez_g[ijk] + Ez_g[ijkm1]);
            float Hx  = 0.25f * (Hx_g[ijk] + Hx_g[ijm1k] + Hx_g[ijkm1] + Hx_g[ijm1km1]);
            float Hy  = 0.25f * (Hy_g[ijk] + Hy_g[im1jk] + Hy_g[ijkm1] + Hy_g[im1jkm1]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]); 

            Sz0x_g[idx + x_width * idy]+= Ey * Hz - Ez * Hy;
            Sz0y_g[idx + x_width * idy]+= Ez * Hx - Ex * Hz;
            Sz0z_g[idx + x_width * idy]+= Ex * Hy - Ey * Hx;
        }
        if(idz == z_width - 2){
            int im1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ijm1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijkm1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);  
            int im1jm1k= idx-1 + x_width * (idy-1)   + (x_width * y_width) *  idz;
            int im1jkm1= idx-1 + x_width * idy       + (x_width * y_width) *  (idz -1);
            int ijm1km1= idx   + x_width * (idy-1)   + (x_width * y_width) *  (idz -1);
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  0.5f * (Ez_g[ijk] + Ez_g[ijkm1]);
            float Hx  = 0.25f * (Hx_g[ijk] + Hx_g[ijm1k] + Hx_g[ijkm1] + Hx_g[ijm1km1]);
            float Hy  = 0.25f * (Hy_g[ijk] + Hy_g[im1jk] + Hy_g[ijkm1] + Hy_g[im1jkm1]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]); 

            SzNx_g[idx + x_width * idy]+= Ey * Hz - Ez * Hy;
            SzNy_g[idx + x_width * idy]+= Ez * Hx - Ex * Hz;
            SzNz_g[idx + x_width * idy]+= Ex * Hy - Ey * Hx;
        }
        if(idz == z_width/2){
            int im1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ijm1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijkm1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);  
            int im1jm1k= idx-1 + x_width * (idy-1)   + (x_width * y_width) *  idz;
            int im1jkm1= idx-1 + x_width * idy       + (x_width * y_width) *  (idz -1);
            int ijm1km1= idx   + x_width * (idy-1)   + (x_width * y_width) *  (idz -1);
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  0.5f * (Ez_g[ijk] + Ez_g[ijkm1]);
            float Hx  = 0.25f * (Hx_g[ijk] + Hx_g[ijm1k] + Hx_g[ijkm1] + Hx_g[ijm1km1]);
            float Hy  = 0.25f * (Hy_g[ijk] + Hy_g[im1jk] + Hy_g[ijkm1] + Hy_g[im1jkm1]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]); 

            SzIx_g[idx + x_width * idy]+= Ey * Hz - Ez * Hy;
            SzIy_g[idx + x_width * idy]+= Ez * Hx - Ex * Hz;
            SzIz_g[idx + x_width * idy]+= Ex * Hy - Ey * Hx;
        }
    }
    }
}
__global__ void Diagnostic_3D(  const float * __restrict__ Ex_g, 
                                const float * __restrict__ Ey_g, 
                                const float * __restrict__ Ez_g, 
                                const float * __restrict__ Hx_g, 
                                const float * __restrict__ Hy_g, 
                                const float * __restrict__ Hz_g,
                                const float * __restrict__ Jx_g,
                                const float * __restrict__ Jy_g,
                                const float * __restrict__ Jz_g,
                                      float * __restrict__ Sx_g, 
                                      float * __restrict__ Sy_g, 
                                      float * __restrict__ Sz_g)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    //int _x_width = blockDim.x * gridDim.x;

    if((idy > 0) && (idx > 0)){
    #if _3D
    for(int idz = 1; idz < z_width; idz++) // loop over z coordinate (numpy axis 0)
    #endif
    {
        #if _3D
        //    int ijk = idx +     x_width * idy + (x_width * y_width) *  idz;
        #else
        //    int ijk = idx +     x_width * idy;
        #endif

        #if _3D
            int ijk  = idx   + x_width * idy   + (x_width * y_width) *  idz;
            int im1jk= idx-1 + x_width * idy   + (x_width * y_width) *  idz;
            int ijm1k= idx   + x_width *(idy-1)+ (x_width * y_width) *  idz;
            int ijkm1= idx   + x_width * idy   + (x_width * y_width) *  (idz - 1);
  
            int im1jm1k= idx-1 + x_width * (idy-1)   + (x_width * y_width) *  idz;
            int im1jkm1= idx-1 + x_width * idy       + (x_width * y_width) *  (idz -1);
            int ijm1km1= idx   + x_width * (idy-1)   + (x_width * y_width) *  (idz -1);
        #else            
            int ijk  = idx   + x_width * idy;
            int im1jk= idx-1 + x_width * idy;
            int ijm1k= idx   + x_width *(idy-1);  
            int im1jm1k= idx-1 + x_width * (idy-1);
        #endif
        
        //Sx_g[ijk] += Ey_g[ijk] * Hz_g[ijk] - Ez_g[ijk] * Hy_g[ijk];
        //Sy_g[ijk] += Ez_g[ijk] * Hx_g[ijk] - Ex_g[ijk] * Hz_g[ijk];
        //Sz_g[ijk] += Ex_g[ijk] * Hy_g[ijk] - Ey_g[ijk] * Hx_g[ijk];

        //if((idx > 0)){
        //    Sx_g[ijk] +=                      (- Ez_g[ijk] * 0.25 * (Hy_g[ijk] + Hy_g[i1jk] + Hy_g[ij1k] + Hy_g[i1j1k]));
        //    Sx_g[ijk] += Ey_g[ijk] * Hz_g[ijk] - Ez_g[ijk] * Hy_g[ijk];
        //}
        //if((idy > 0)){
        //    Sy_g[ijk] +=    Ez_g[ijk] * 0.5 * (Hx_g[ijk] + Hx_g[ij1k]);
        //}
        //Sz_g[ijk]  =   0.0;//+= Ez * Ez;
        
        #if _3D
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  0.5f * (Ez_g[ijk] + Ez_g[ijkm1]);
            float Hx  = 0.25f * (Hx_g[ijk] + Hx_g[ijm1k] + Hx_g[ijkm1] + Hx_g[ijm1km1]);
            float Hy  = 0.25f * (Hy_g[ijk] + Hy_g[im1jk] + Hy_g[ijkm1] + Hy_g[im1jkm1]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]);        
            Sx_g[ijk] += Ey * Hz - Ez * Hy;
            Sy_g[ijk] += Ez * Hx - Ex * Hz;
            Sz_g[ijk] += Ex * Hy - Ey * Hx;
        #else
            float Ex  =  0.5f * (Ex_g[ijk] + Ex_g[im1jk]);
            float Ey  =  0.5f * (Ey_g[ijk] + Ey_g[ijm1k]);
            float Ez  =  Ez_g[ijk];
            float Hx  = 0.5f * (Hx_g[ijk] + Hx_g[ijm1k]);
            float Hy  = 0.5f * (Hy_g[ijk] + Hy_g[im1jk]);
            float Hz  = 0.25f * (Hz_g[ijk] + Hz_g[ijm1k] + Hz_g[im1jm1k] + Hz_g[im1jk]);        
            Sx_g[ijk] += Ey * Hz - Ez * Hy;
            Sy_g[ijk] += Ez * Hx - Ex * Hz;
            Sz_g[ijk] += Ex * Hy - Ey * Hx;
        #endif
    }
    }
}

__global__ void Phasor_3D(int t,
                          int location,
                       float n_steps_per_lambda,
                       float z0,
                       float w_0,
                       float x_sofs,
                       float k_x,
                       float k_z,
                       float E0_x,
                       float phy0_x,
                       float E0_y,
                       float phy0_y,
                       float E0_z,
                       float phy0_z,
                       const float * __restrict__ Ex_g, 
                       const float * __restrict__ Ey_g, 
                       const float * __restrict__ Ez_g, 
                             float * __restrict__ reAx_g, 
                             float * __restrict__ reAy_g, 
                             float * __restrict__ reAz_g,
                             float * __restrict__ imAx_g, 
                             float * __restrict__ imAy_g, 
                             float * __restrict__ imAz_g)
                       
{
    int   idx     = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    //int   x_width = blockDim.x * gridDim.x;

    #if _3D
    int   idy_z   = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    if((idy_z > 0) && (idy_z < z_width - 2))
    #endif
    {
        #if _3D
            //int ijk = idx + x_width * (_nPML + 0) + (x_width * y_width) * idy_z;
            int ijk; int width;
            switch(location){
            default:      ijk = idx + x_width * _nPML;                width = x_width; break; //y0 
            case 2:       ijk = _nPML + x_width * idx;                width = y_width; break; //x0
            case 3:       ijk = idx + x_width * (y_width - 1 - _nPML);width = x_width; break; //yN
            case 4:       ijk = (x_width - 1 - _nPML) + x_width * idx;width = y_width; break; //xN
            }
            ijk+= (x_width * y_width) * idy_z;
        #else
            int ijk; int width;
            switch(location){
                default:      ijk = idx + x_width * _nPML;                width = x_width; break; //y0 
                case 2:       ijk = _nPML + x_width * idx;                width = y_width; break; //x0
                case 3:       ijk = idx + x_width * (y_width - 1 - _nPML);width = x_width; break; //yN
                case 4:       ijk = (x_width - 1 - _nPML) + x_width * idx;width = y_width; break; //xN
            }
        #endif
        if((idx > _nPML) && (idx < width - _nPML)){
            float X_box   = float(idx-x_sofs);
            #if _3D
                float Z_box   = float(idy_z-z_width/2);
                float z   = (X_box*k_x+Z_box*k_z)+z0;
                float r   = sqrtf(powf(X_box-k_x*(X_box*k_x+Z_box*k_z),2)
                                 +(1.f-k_x*k_x-k_z*k_z)*powf((X_box*k_x+Z_box*k_z),2)
                                 +powf(Z_box-k_z*(X_box*k_x+Z_box*k_z),2));
            #else
                float r   = fabsf(X_box*sqrtf(1.f-k_x*k_x));
                float z   = (X_box*k_x)+z0;
            #endif
                float z_R = CUDART_PI_F * w_0 * w_0 / n_steps_per_lambda;  
                float w   = w_0 * sqrtf(1.f+z*z/(z_R*z_R));

                float GBeam_Amplitude = (w_0/w) * expf(-fabsf(powf(r,2))/powf(w,2));
                float R   = z * (1.f + z_R * z_R / (z*z));
                
                float sin;
                float cos;
                sincospif(-phy0_x + 2.0f * (_Courant * t/1.f + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda, &sin, &cos);
                reAx_g[idx] = GBeam_Amplitude * E0_x * Ex_g[ijk] * cos;
                imAx_g[idx] = GBeam_Amplitude * E0_x * Ex_g[ijk] * sin;

                sincospif(-phy0_y + 2.0f * (_Courant * t/1.f + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda, &sin, &cos);
                reAy_g[idx] = GBeam_Amplitude * E0_y * Ey_g[ijk] * cos;
                imAy_g[idx] = GBeam_Amplitude * E0_y * Ey_g[ijk] * sin;
                
                #if _Full2D
                sincospif(-phy0_z + 2.0f * (_Courant * t/1.f + z + 0.5f * powf(r,2) / R) / n_steps_per_lambda, &sin, &cos);
                reAz_g[idx] = GBeam_Amplitude * E0_z * Ez_g[ijk] * cos;
                imAz_g[idx] = GBeam_Amplitude * E0_z * Ez_g[ijk] * sin;
                #endif
               
        }
    }
}
__global__ void Phasor_3D_old(        int t,
                                float n_steps_per_lambda,
                                const float * __restrict__ Ex_g, 
                                const float * __restrict__ Ey_g, 
                                const float * __restrict__ Ez_g, 
                                      float * __restrict__ reAx_g, 
                                      float * __restrict__ reAy_g, 
                                      float * __restrict__ reAz_g,
                                      float * __restrict__ imAx_g, 
                                      float * __restrict__ imAy_g, 
                                      float * __restrict__ imAz_g)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    //int _x_width = blockDim.x * gridDim.x;

    #if _3D
    for(int idz = 0; idz < z_width; idz++) // loop over z coordinate (numpy axis 0)
    #endif
    {
        #if _3D
            //int y_width = blockDim.y * gridDim.y;
            int ijk = idx +     x_width * idy + (x_width * y_width) *  idz;
        #else
            int ijk = idx +     x_width * idy;
        #endif

        float sin;
        float cos;
        sincospif(2.0f * (-_Courant * t ) / n_steps_per_lambda, &sin, &cos);

        reAx_g[ijk] += Ex_g[ijk] * cos;
        imAx_g[ijk] += Ex_g[ijk] * sin;

        reAy_g[ijk] += Ey_g[ijk] * cos;
        imAy_g[ijk] += Ey_g[ijk] * sin;

        reAz_g[ijk] += Ez_g[ijk] * cos;
        imAz_g[ijk] += Ez_g[ijk] * sin;
    }
}

__global__ void Window_batch(      float * __restrict__ Bx_g, 
                             const float * __restrict__ Ax_g, 
                                                    int z_fft,
                                                    int z_min,
                                                    int z_max,
                                                    int W_width)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // x coordinate (numpy axis 2)
    int idy = threadIdx.y + blockIdx.y * blockDim.y; // y coordinate (numpy axis 1)
    //int x_width = blockDim.x * gridDim.x;
    //int y_width = blockDim.y * gridDim.y;
    #if _3D
        int ijkA = idx +     x_width * idy + (x_width * y_width) *  z_fft;
    #else
        int ijkA = idx +     x_width * idy;
    #endif
    float Ax=Ax_g[ijkA];

    for(int idz = z_min; idz < z_max; idz++) // loop over z coordinate (numpy axis 0)
    {
        int ijkB = idx +     x_width * idy + (x_width * y_width) *  (idz-z_min);
        float l = fminf(fmaxf((1.f*idy-idz)/W_width+0.5f,0.f),1.f);
        Bx_g[ijkB]=Ax*powf(sinpif(l),2);
    }
}
