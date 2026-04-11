import numpy as np
from scipy.special import spherical_jn, spherical_yn, lpmv

def mask_sphere(data, r0, grid_size):
    """
    Set field values inside the sphere to zero.

    Args:
        data: [resolution, resolution, resolution] complex field values
        r0: sphere radius
        grid_size: domain grid extent (grid spans -grid_size/2 to +grid_size/2)

    Returns:
        Masked data with interior points set to zero
    """
    resolution = data.shape[0]
    x = np.linspace(-grid_size / 2, grid_size / 2, resolution)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # Create mask: True outside sphere, False inside
    mask = R >= r0

    return data * mask


def sphere_analytic(k0, r0, incident_direction, grid_size, resolution, n_terms=30):
    """
    Compute analytic scattered field from a rigid (sound-hard) sphere.

    Uses the Mie series solution for acoustic scattering. The incident field is
    a plane wave p_inc = exp(i*k*r*cos(theta)) traveling in the incident_direction.

    For a rigid sphere (Neumann BC: dp/dn = 0), the scattered field is:
        p_s = -sum_{n=0}^{inf} (2n+1) * i^n * (j'_n(ka)/h'_n(ka)) * h_n(kr) * P_n(cos(theta))

    where j_n, h_n are spherical Bessel/Hankel functions, P_n are Legendre polynomials,
    and theta is the angle from the incident direction.

    Args:
        k0: wavenumber
        r0: sphere radius
        incident_direction: [3] unit vector for incident wave direction
        grid_size: domain grid extent
        resolution: grid resolution per axis
        n_terms: number of terms in Mie series (default 30)

    Returns:
        [resolution, resolution, resolution] complex scattered field
    """
    # Normalize incident direction
    incident_direction = np.array(incident_direction)
    incident_direction = incident_direction / np.linalg.norm(incident_direction)

    # Create domain grid matching JAX_BEM_kirchoff_helmholtz.create_domain_grid
    x = np.linspace(-grid_size / 2, grid_size / 2, resolution)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Compute spherical coordinates relative to incident direction
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # cos(theta) = dot(position, incident_direction) / |position|
    # This is the angle between each point and the incident wave direction
    cos_theta = (X * incident_direction[0] +
                 Y * incident_direction[1] +
                 Z * incident_direction[2])
    # Avoid division by zero at origin
    cos_theta = np.where(R > 1e-10, cos_theta / R, 0.0)

    # Precompute k*a (wavenumber * radius) for coefficients
    ka = k0 * r0

    # Initialize scattered field
    p_scattered = np.zeros((resolution, resolution, resolution), dtype=complex)

    # Compute Mie series coefficients for rigid sphere
    # For Neumann BC: coefficient A_n = -j'_n(ka) / h'_n(ka)
    for n in range(n_terms):
        # Spherical Bessel functions at ka
        jn_ka = spherical_jn(n, ka)
        yn_ka = spherical_yn(n, ka)
        hn_ka = jn_ka + 1j * yn_ka  # Spherical Hankel function of first kind

        # Derivatives: j'_n(x) = j_{n-1}(x) - (n+1)/x * j_n(x)
        # Using recurrence: j'_n(x) = (n/x)*j_n(x) - j_{n+1}(x)
        jn_ka_deriv = spherical_jn(n, ka, derivative=True)
        yn_ka_deriv = spherical_yn(n, ka, derivative=True)
        hn_ka_deriv = jn_ka_deriv + 1j * yn_ka_deriv

        # Scattering coefficient for rigid sphere
        A_n = -jn_ka_deriv / hn_ka_deriv

        # Evaluate spherical Hankel function at all field points
        kr = k0 * R
        # Handle small kr values to avoid singularities
        kr_safe = np.where(kr > 1e-10, kr, 1e-10)

        jn_kr = spherical_jn(n, kr_safe)
        yn_kr = spherical_yn(n, kr_safe)
        hn_kr = jn_kr + 1j * yn_kr

        # Legendre polynomial P_n(cos_theta)
        Pn = lpmv(0, n, cos_theta)  # m=0 gives P_n

        # Add contribution: (2n+1) * i^n * A_n * h_n(kr) * P_n(cos_theta)
        coeff = (2 * n + 1) * (1j ** n) * A_n
        p_scattered += coeff * hn_kr * Pn

    # Set interior points to zero (field is only valid outside sphere)
    p_scattered = mask_sphere(p_scattered, r0, grid_size)

    return p_scattered
