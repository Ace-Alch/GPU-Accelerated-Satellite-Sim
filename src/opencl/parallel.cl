/*  Copyright (c) 2016 
                      Matias Koskela:       matias.koskela@tut.fi
                      Heikki Kultala:       heikki.kultala@tut.fi
                      Topi Leppanen:        topi.leppanen@tuni.fi
                      Mehdi Moallemkolaei:  Mehdi.moallemkolaei@tuni.fi
                      Ashfak Nehal:         MdAshfakHaider.nehal@tuni.fi
*/

__kernel void shade(
    // --global makes mamory shared between multi threads to read/write
    __global uchar4* k_out_pixels,        // a vector (1D Array) of 4 unsigned bytes (B, G, R, A)
    __global const float* k_sat_pos_x,    // SoA Sat Pos X
    __global const float* k_sat_pos_y,    // SoA Sat Pos Y
    __global const float* k_id_r,         // SoA Sat R
    __global const float* k_id_g,         // SoA Sat G
    __global const float* k_id_b,         // SoA Sat B
    const int             k_sat_count,    // SAT Count
    const int             k_width,        // Image Size X
    const int             k_height,       // Image Size Y
    const float           k_bh_r2,        // BLACK_HOLE_RADIUS^2
    const float           k_sat_r2,       // SATELLITE_RADIUS^2
    const int             k_mouse_x,      // black hole center X
    const int             k_mouse_y)      // black hole center Y
{

    // each thread, shades one pixel with k_x and k_y:
    const int   k_x = get_global_id(0); //global X index of a work-item
    const int   k_y = get_global_id(1); //global Y index of a work-item

    // as we round up the window to wg size in .c , now some threads are outside the window, so must be exited.
    if (k_x >= k_width || k_y >= k_height) return;

    // k_out_pixels is 1D, but the image is 2D, so we make y=ax+b to make linear y (k_idx)
    const int   k_idx = k_y * k_width + k_x;
    const float k_px = (float)k_x;
    const float k_py = (float)k_y;

    // black hole check (no sqrt) - coloring black
    float k_dxBH = k_px - (float)k_mouse_x; 
    float k_dyBH = k_py - (float)k_mouse_y;
    float k_d2BH = k_dxBH * k_dxBH + k_dyBH * k_dyBH; // my pixel's distance to bh

    if (k_d2BH < k_bh_r2) {
        k_out_pixels[k_idx] = (uchar4)(0, 0, 0, 0);   // BGRA = black
        return;
    }

    float k_sumR = 0.0f, k_sumG = 0.0f, k_sumB = 0.0f;  // Sum of (satellite_color * weight) over all satellites
    float k_weights = 0.0f; // Sum of all weights
    float k_shortestD2 = INFINITY;  // Track the closest satellite
    float k_nR = 0.0f, k_nG = 0.0f, k_nB = 0.0f;      // Color of the nearest satellite
    int   k_hit = 0;    // hit Flag

    // Single-pass satellite loop - same logic as BH shading
    for (int k_j = 0; k_j < k_sat_count; ++k_j) {
        float k_dx = k_px - k_sat_pos_x[k_j];
        float k_dy = k_py - k_sat_pos_y[k_j];
        float k_d2 = k_dx * k_dx + k_dy * k_dy; // pixel's distance to any satellite
        
        // Satellite Coloring:
        // if inside a satellite:
        if (k_d2 < k_sat_r2) {
            k_out_pixels[k_idx] = (uchar4)(255, 255, 255, 0); // BGRA = white
            k_hit = 1;
            break; // eaten by BH
        }
        
        // Space Coloring:
        // if NOT inside a satellite,
        // then closer satellites should influence the pixel more
        // and farther ones influence less...:
        float k_inv = 1.0f / k_d2;
        float k_w = k_inv * k_inv;  // w=1/d^4 -> to reduce the far satellite's effect
        
        // accumulating
        k_weights += k_w;

        // weighted coloring: Color � Weight
        k_sumR += k_id_r[k_j] * k_w;
        k_sumG += k_id_g[k_j] * k_w;
        k_sumB += k_id_b[k_j] * k_w;

        // if new satellite comes closer than earlier sat, then UPDATE:
        if (k_d2 < k_shortestD2) {
            k_shortestD2 = k_d2;    // UPDATE Satellite
            // store its colors
            k_nR = k_id_r[k_j];
            k_nG = k_id_g[k_j];
            k_nB = k_id_b[k_j];
        }
    }

    if (!k_hit) {
        float k_invW = 1.0f / k_weights;
        float k_r = k_nR + 3.0f * (k_sumR * k_invW);
        float k_g = k_nG + 3.0f * (k_sumG * k_invW);
        float k_b = k_nB + 3.0f * (k_sumB * k_invW);

        // Convert to BGRA 0..255
        uchar k_ur = (uchar)(k_r * 255.0f);
        uchar k_ug = (uchar)(k_g * 255.0f);
        uchar k_ub = (uchar)(k_b * 255.0f);
        k_out_pixels[k_idx] = (uchar4)(k_ub, k_ug, k_ur, (uchar)0);
    }
}
