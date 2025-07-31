import numpy as np

# Parameters we fix
m_star = 0.023 # effective mass of InAs
m_0 = 9.1093837*1e-31 # kg
Delta_Al = 0.34 # meV   
#Delta_Al = 0.5
hbar_mev = 6.582119569 * 1e-13 #meV*s
hbar_Js = 1.05457182 * 1e-34 # J * s


def S_AC(w, phi, n, l_S, gamma, T, E_F, l_S1 = 0., l_Sn = 0.):
    ''' Scattering matrix for Andreev Crystal with n superconductors '''
    # Calculate parameters
    k_F = np.sqrt(2*E_F*m_star*m_0/(hbar_mev*hbar_Js)) # 1/m
    v_F = hbar_Js*k_F / (m_star*m_0) # m/s
    xi_0 = hbar_mev * v_F / (np.pi*gamma*Delta_Al) # m
    L_S = l_S#/xi_0 # l_S/xi_0
    k_FS = k_F*l_S*xi_0 # k_F*l_S #when l_S = L_S/xi_0
    # L_N = l_N/xi_0 # l_N / xi_0
    # k_FN = k_F*l_N # k_F*l_N
    L_S1 = l_S1#/xi_0
    L_Sn = l_Sn#/xi_0
    #print(f'L_S: {L_S}, L_S1: {L_S1}, L_Sn: {L_Sn}')

    # Define concatination of matrices
    def concatenate(S1, S2):
        ''' Function to concatinate two scattering matrices S1 and S2 '''
        # Reflection and transmission matrices of S1
        rL1 = S1[0:2,0:2]
        rR1 = S1[2:4,2:4]
        tLR1 = S1[0:2,2:4]
        tRL1 = S1[2:4,0:2]

        # Reflection and transmission matrices of S2
        rL2 = S2[0:2,0:2]
        rR2 = S2[2:4,2:4]
        tLR2 = S2[0:2,2:4]
        tRL2 = S2[2:4,0:2]

        # Calculate the reflection and transmission of the concatenated scattering matrix
        rL12 = rL1 + tLR1 @ rL2 @ np.linalg.inv( np.identity(2) - rR1 @ rL2 ) @ tRL1
        rR12 = rR2 + tRL2 @ np.linalg.inv( np.identity(2) - rR1 @ rL2 ) @ rR1 @ tLR2
        tRL12 = tRL2 @ np.linalg.inv(np.identity(2) - rR1 @ rL2) @ tRL1
        tLR12 = tLR1 @ ( np.identity(2) + rL2 @ np.linalg.inv(np.identity(2) - rR1 @ rL2) @ rR1 ) @ tLR2

        # Final scattering matrix
        Scon = np.zeros((4,4), dtype='complex')
        Scon[0:2,0:2] = rL12
        Scon[2:4,2:4] = rR12
        Scon[0:2,2:4] = tLR12
        Scon[2:4,0:2] = tRL12

        return Scon
    
    # Selfenergy of superconductor
    def w_selfenergy(E):
        ''' Selfenergy of superconductor '''
        return E* ( 1 + np.sqrt(1-E**2+0j)/gamma )
    
    # # Define ke and kh
    # def ke(E):
    #     return k_FN + E*L_N
    # def kh(E):
    #     return k_FN - E*L_N
    
    def r_he(phi, E, L_S = L_S):
        ''' electron to hole reflection amplitude '''
        exp_2iqls = np.exp(-2*np.pi*L_S)
        # if np.abs(E) > 1:
        #     r_blonder = (E - np.sqrt(E**2 - 1) + 0j)
        # else:
        #     r_blonder = np.exp(-1j*np.arccos(E + 0j))
        if E > 1:
            r_blonder = (E - np.sqrt(E**2 - 1) + 0j)
        elif E < -1:
            r_blonder = (E + np.sqrt(E**2 - 1) + 0j)
        else:
            r_blonder = np.exp(-1j*np.arccos(E + 0j))
        
        return (1-exp_2iqls)*r_blonder*np.exp(-1j*phi) / (1 - exp_2iqls*r_blonder*r_blonder)
    
    def r_eh(phi, E, L_S = L_S):
        ''' hole to electron reflection amplitude '''
        return -np.conjugate(r_he(phi, -np.conjugate(E), L_S))
    
    def t_ee(E, L_S = L_S):
        ''' electron to electron transmission amplitude '''
        exp_iqls = np.exp(-np.pi*L_S)
        exp_2iqls = np.exp(-2*np.pi*L_S)
        
        # if np.abs(E) > 1:
        #     r_blonder = (E - np.sqrt(E**2 - 1) + 0j)
        # else:
        #     r_blonder = np.exp(-1j*np.arccos(E + 0j))
        if E > 1:
            r_blonder = (E - np.sqrt(E**2 - 1) + 0j)
        elif E < -1:
            r_blonder = (E + np.sqrt(E**2 - 1) + 0j)
        else:
            r_blonder = np.exp(-1j*np.arccos(E + 0j))

        return (1-r_blonder*r_blonder)*exp_iqls*np.exp(1j*k_FS)/ (1 - exp_2iqls*r_blonder*r_blonder)

    def t_hh(E, L_S = L_S):
        ''' hole to hole transmission amplitude '''
        return np.conjugate(t_ee(-np.conjugate(E), L_S))
    
    def S_NSN(phi,w, L_S = L_S):
        ''' Scattering matrix for NSN junction '''
        reh = r_eh(phi,w_selfenergy(w), L_S)
        rhe = r_he(phi,w_selfenergy(w), L_S)
        tee = t_ee(w_selfenergy(w), L_S)
        thh = t_hh(w_selfenergy(w), L_S)

        S = np.array( [ [0, reh, tee, 0],
                      [rhe, 0, 0, thh],
                      [tee, 0, 0, reh],
                      [0, thh, rhe, 0] ] )
        return S
    
    def S_NNN(T):
        r = np.sqrt(1-T)
        t = np.sqrt(T)

        S = np.array( [ [r, 0, t, 0],
                        [0, r, 0, t],
                        [t, 0, -r, 0],
                        [0, t, 0, -r] ] )
        return S
    
    # Calculate scattering matrix for n superconductors
    for i in range(n):
        if i == 0:
            # Scattering matrix for first superconductor
            S = S_NSN(0, w)
        else:
            S = concatenate(S, S_NNN(T))
            S = concatenate(S, S_NSN(i*phi, w))
    
    # Attach first and last superconducor if nonzero
    if L_S1 != 0:
        S = concatenate(S_NNN(T), S)
        S = concatenate(S_NSN(-phi, w, L_S1), S)
    if L_Sn != 0:
        S = concatenate(S, S_NNN(T))
        S = concatenate(S, S_NSN(n*phi, w, L_Sn))
    
    return S

def Gmat(S):
    GLL = 1 - (np.abs( S[0,0] )**2 - np.abs( S[1,0] )**2 )# + np.abs(S[0,1])**2 - np.abs(S[1,1])**2)
    GLR = -(np.abs(S[0,2])**2 - np.abs(S[1,2])**2 )# + np.abs(S[0,3])**2 - np.abs(S[1,3])**2)
    GRL = -(np.abs(S[2,0])**2 - np.abs(S[3,0])**2 )# + np.abs(S[2,1])**2 - np.abs(S[3,1])**2)
    GRR = 1 - (np.abs(S[2,2])**2 - np.abs(S[3,2])**2 )# + np.abs(S[2,3])**2 - np.abs(S[3,3])**2)
    return GLL, GLR, GRL, GRR

def G(Elst, num_E, philst, num_phi, n, l_S, gamma, T, E_F, L_S1 = 0, L_Sn = 0):
    GLL = np.zeros((num_E, num_phi))
    GLR = np.zeros((num_E, num_phi))
    GRL = np.zeros((num_E, num_phi))
    GRR = np.zeros((num_E, num_phi))
    for i, E in enumerate(Elst):
        for j, phi in enumerate(philst):
            S = S_AC(E, phi, n, l_S, gamma, T, E_F, L_S1, L_Sn)
            GLL[i,j], GLR[i,j], GRL[i,j], GRR[i,j] = Gmat(S)
    return GLL, GLR, GRL, GRR

def G_1d(Elst, phi, num, n, l_S, gamma, T, E_F, L_S1 = 0, L_Sn = 0):
    GLL = np.zeros(num)
    GLR = np.zeros(num)
    GRL = np.zeros(num)
    GRR = np.zeros(num)
    for i, E in enumerate(Elst):
        S = S_AC(E, phi, n, l_S, gamma, T, E_F, L_S1, L_Sn)
        GLL[i], GLR[i], GRL[i], GRR[i] = Gmat(S)
    return GLL, GLR, GRL, GRR