#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY Gaussian Splat Loader
Unified loader for both standard PLY and SuperSplat compressed PLY formats
Extracted and rewritten from SuperSplat project for standalone usage
"""

import numpy as np
import struct
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from plyfile import PlyData


SH_C0 = 0.28209479177387814  # Y_0^0 spherical harmonic coefficient
SQRT2_INV = 1.0 / np.sqrt(2.0)


@dataclass
class GaussianSplatData:
    """Data structure to hold Gaussian Splat information"""
    # Basic properties
    positions: np.ndarray  # (N, 3) - x, y, z coordinates
    rotations: np.ndarray  # (N, 4) - quaternion [w, x, y, z] or [x, y, z, w]
    scales: np.ndarray     # (N, 3) - log scale values
    colors_dc: np.ndarray  # (N, 3) - diffuse color DC coefficients
    opacities: np.ndarray  # (N, 1) - opacity values (logit space)

    # Optional spherical harmonics (excluding DC)
    sh_rest: Optional[np.ndarray] = None  # (N, 45) for 3 bands excluding DC

    # Metadata
    num_splats: int = 0
    sh_bands: int = 0
    is_compressed: bool = False

    def __post_init__(self):
        self.num_splats = len(self.positions)
        if self.sh_rest is not None:
            # Calculate number of SH bands from coefficients
            # 0 bands: 0 coeffs, 1 band: 3 coeffs, 2 bands: 8 coeffs, 3 bands: 15 coeffs per channel
            coeffs_per_channel = self.sh_rest.shape[1] // 3
            if coeffs_per_channel == 0:
                self.sh_bands = 0
            elif coeffs_per_channel == 3:
                self.sh_bands = 1
            elif coeffs_per_channel == 8:
                self.sh_bands = 2
            elif coeffs_per_channel == 15:
                self.sh_bands = 3
            else:
                self.sh_bands = 0  # Unknown format
        else:
            self.sh_bands = 0

    def get_linear_colors(self) -> np.ndarray:
        """
        Convert DC coefficients to linear RGB colors

        Returns:
            np.ndarray: Linear RGB colors (N, 3), range [0,1]
        """
        rgb_linear = np.clip(self.colors_dc * SH_C0 + 0.5, 0.0, 1.0)
        return rgb_linear.astype(np.float32)

    def get_rotations_xyzw(self) -> np.ndarray:
        """
        Convert rotations from WXYZ to XYZW format for gsplat compatibility

        Returns:
            np.ndarray: Normalized quaternions in XYZW format (N, 4)
        """
        # Convert from WXYZ to XYZW format
        rotations_xyzw = self.rotations[:, [3, 1, 2, 0]]

        # Normalize quaternions
        norms = np.linalg.norm(rotations_xyzw, axis=1, keepdims=True)
        rotations_xyzw = rotations_xyzw / np.maximum(norms, 1e-8)

        return rotations_xyzw.astype(np.float32)

    def get_sh_coefficients(self) -> Optional[np.ndarray]:
        """
        Return SH coefficients for rendering (DC + rest), shape (N, K+1, 3):
          - Index 0 is DC coefficients (three channels)
          - Indices 1..K are non-DC coefficients per channel, grouped by degree

        Note: SuperSplat compressed PLY stores f_rest_* in channel-major order:
          [R0..R(K-1), G0..G(K-1), B0..B(K-1)]
        This method reorganizes them into (K, 3) format: [(R_d, G_d, B_d) for d in 0..K-1]
        """
        if self.sh_rest is None:
            return None

        N = self.num_splats
        total = self.sh_rest.shape[1]
        if total % 3 != 0:
            return None

        K = total // 3  # Non-DC coefficients per channel (1 band=3, 2 bands=8, 3 bands=15)

        # Reorganize channel-major storage: [R(0..K-1), G(0..K-1), B(0..K-1)] -> (N, K, 3)
        r = self.sh_rest[:, 0:K]
        g = self.sh_rest[:, K:2*K]
        b = self.sh_rest[:, 2*K:3*K]
        sh_rest_reshaped = np.stack([r, g, b], axis=2).astype(np.float32)  # (N, K, 3)

        # DC coefficients go first (N, 1, 3)
        dc_coeffs = self.colors_dc.reshape(N, 1, 3).astype(np.float32)

        # Combine to get (N, K+1, 3)
        sh_coefficients = np.concatenate([dc_coeffs, sh_rest_reshaped], axis=1)
        return sh_coefficients


    def print_info(self):
        """Print basic information about the loaded data"""
        print(f"Gaussian Splat Data Info:")
        print(f"  Number of splats: {self.num_splats}")
        print(f"  SH bands: {self.sh_bands}")
        print(f"  Is compressed format: {self.is_compressed}")
        print(f"  Data shapes:")
        print(f"    Positions: {self.positions.shape}")
        print(f"    Rotations: {self.rotations.shape}")
        print(f"    Scales: {self.scales.shape}")
        print(f"    Colors DC: {self.colors_dc.shape}")
        print(f"    Opacities: {self.opacities.shape}")
        if self.sh_rest is not None:
            print(f"    SH coefficients: {self.sh_rest.shape}")
        print(f"  Data ranges:")
        print(f"    Position range: [{self.positions.min():.3f}, {self.positions.max():.3f}]")
        print(f"    Scale range: [{self.scales.min():.3f}, {self.scales.max():.3f}]")
        print(f"    Opacity range: [{self.opacities.min():.3f}, {self.opacities.max():.3f}]")

        # Display color information
        linear_colors = self.get_linear_colors()
        print(f"  Color info:")
        print(f"    DC range: [{self.colors_dc.min():.3f}, {self.colors_dc.max():.3f}]")
        print(f"    Linear RGB range: [{linear_colors.min():.3f}, {linear_colors.max():.3f}]")


class ChunkData:
    """Helper class to handle SuperSplat compressed chunk data"""

    def __init__(self, chunk_element):
        self.min_x = np.asarray(chunk_element['min_x'], dtype=np.float32)
        self.min_y = np.asarray(chunk_element['min_y'], dtype=np.float32)
        self.min_z = np.asarray(chunk_element['min_z'], dtype=np.float32)
        self.max_x = np.asarray(chunk_element['max_x'], dtype=np.float32)
        self.max_y = np.asarray(chunk_element['max_y'], dtype=np.float32)
        self.max_z = np.asarray(chunk_element['max_z'], dtype=np.float32)

        self.min_scale_x = np.asarray(chunk_element['min_scale_x'], dtype=np.float32)
        self.min_scale_y = np.asarray(chunk_element['min_scale_y'], dtype=np.float32)
        self.min_scale_z = np.asarray(chunk_element['min_scale_z'], dtype=np.float32)
        self.max_scale_x = np.asarray(chunk_element['max_scale_x'], dtype=np.float32)
        self.max_scale_y = np.asarray(chunk_element['max_scale_y'], dtype=np.float32)
        self.max_scale_z = np.asarray(chunk_element['max_scale_z'], dtype=np.float32)

        self.min_r = np.asarray(chunk_element['min_r'], dtype=np.float32)
        self.min_g = np.asarray(chunk_element['min_g'], dtype=np.float32)
        self.min_b = np.asarray(chunk_element['min_b'], dtype=np.float32)
        self.max_r = np.asarray(chunk_element['max_r'], dtype=np.float32)
        self.max_g = np.asarray(chunk_element['max_g'], dtype=np.float32)
        self.max_b = np.asarray(chunk_element['max_b'], dtype=np.float32)

        # Note: SuperSplat does not write SH min/max to chunks,
        # so we don't attempt to load them here

        self.chunk_count = len(self.min_x)
        self.chunk_size = 256  # Standard SuperSplat chunk size


class PLYGaussianLoader:
    """Loader class for PLY Gaussian Splat files"""

    def __init__(self):
        self.chunk_data = None
        self.is_compressed_format = False

    def load_ply(self, ply_path: str) -> GaussianSplatData:
        """
        Load a PLY file containing Gaussian Splat data.
        Automatically detects and handles both standard and compressed formats.

        Args:
            ply_path: Path to the PLY file

        Returns:
            GaussianSplatData object containing loaded data
        """

        with open(ply_path, 'rb') as f:
            ply = PlyData.read(f)

        # Check if this is a compressed SuperSplat PLY
        element_names = [e.name for e in ply.elements]
        self.is_compressed_format = 'chunk' in element_names and 'vertex' in element_names

        if self.is_compressed_format:
            return self._load_compressed_ply(ply)
        else:
            return self._load_standard_ply(ply)

    def _load_standard_ply(self, ply: PlyData) -> GaussianSplatData:
        """Load standard PLY format"""
        vertex_element = ply['vertex']
        num_splats = len(vertex_element)

        # Extract basic properties
        positions = np.column_stack([
            vertex_element['x'].astype(np.float32),
            vertex_element['y'].astype(np.float32),
            vertex_element['z'].astype(np.float32)
        ])

        # Handle different rotation conventions
        if all(prop in vertex_element for prop in ['rot_0', 'rot_1', 'rot_2', 'rot_3']):
            # PlayCanvas convention: rot_0=w, rot_1=x, rot_2=y, rot_3=z
            rotations = np.column_stack([
                vertex_element['rot_0'].astype(np.float32),  # w
                vertex_element['rot_1'].astype(np.float32),  # x
                vertex_element['rot_2'].astype(np.float32),  # y
                vertex_element['rot_3'].astype(np.float32)   # z
            ])
        else:
            # Try alternative naming convention
            rotations = np.column_stack([
                vertex_element['rot_w'].astype(np.float32) if 'rot_w' in vertex_element else np.ones(num_splats),
                vertex_element['rot_x'].astype(np.float32) if 'rot_x' in vertex_element else np.zeros(num_splats),
                vertex_element['rot_y'].astype(np.float32) if 'rot_y' in vertex_element else np.zeros(num_splats),
                vertex_element['rot_z'].astype(np.float32) if 'rot_z' in vertex_element else np.zeros(num_splats)
            ])

        # Normalize quaternions
        norms = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations = rotations / np.maximum(norms, 1e-8)

        # Extract scales (log space)
        scales = np.column_stack([
            vertex_element['scale_0'].astype(np.float32),
            vertex_element['scale_1'].astype(np.float32),
            vertex_element['scale_2'].astype(np.float32)
        ])

        # Extract DC color coefficients
        colors_dc = np.column_stack([
            vertex_element['f_dc_0'].astype(np.float32),
            vertex_element['f_dc_1'].astype(np.float32),
            vertex_element['f_dc_2'].astype(np.float32)
        ])

        # Extract opacity
        opacities = vertex_element['opacity'].astype(np.float32).reshape(-1, 1)

        # Extract spherical harmonics (if present)
        sh_rest = None
        sh_names = [f'f_rest_{i}' for i in range(45)]  # Max 3 bands * 15 coeffs * 3 colors = 45
        available_sh = [name for name in sh_names if name in vertex_element.dtype.names]

        if available_sh:
            sh_coeffs = []
            for name in available_sh:
                sh_coeffs.append(vertex_element[name].astype(np.float32))
            if sh_coeffs:
                sh_rest = np.column_stack(sh_coeffs)

        return GaussianSplatData(
            positions=positions,
            rotations=rotations,
            scales=scales,
            colors_dc=colors_dc,
            opacities=opacities,
            sh_rest=sh_rest,
            is_compressed=False
        )

    def _load_compressed_ply(self, ply: PlyData) -> GaussianSplatData:
        """Load SuperSplat compressed PLY format"""
        # Load chunk data
        if 'chunk' in [e.name for e in ply.elements]:
            chunk_element = ply['chunk']
            self.chunk_data = ChunkData(chunk_element)

        # Load vertex data
        vertex_element = ply['vertex']
        num_splats = len(vertex_element)

        packed_position = np.asarray(vertex_element['packed_position'], dtype=np.uint32)
        packed_rotation = np.asarray(vertex_element['packed_rotation'], dtype=np.uint32)
        packed_scale = np.asarray(vertex_element['packed_scale'], dtype=np.uint32)
        packed_color = np.asarray(vertex_element['packed_color'], dtype=np.uint32)

        # Compute chunk indices (sequential grouping of 256)
        chunk_indices = self._compute_chunk_indices_sequential(num_splats)

        # Unpack data
        positions = self._unpack_position_with_indices(packed_position, chunk_indices)
        scales = self._unpack_scale_with_indices(packed_scale, chunk_indices)
        rotations = self._unpack_rotation_supersplat(packed_rotation)
        colors_dc, opacities = self._unpack_color(packed_color, chunk_indices, order='RGBA', return_logit=False)

        # Load spherical harmonics if present
        sh_rest = None
        if 'sh' in [e.name for e in ply.elements]:
            sh_element = ply['sh']
            sh_rest = self._unpack_sh_coefficients(sh_element, chunk_indices)

        return GaussianSplatData(
            positions=positions,
            rotations=rotations,
            scales=scales,
            colors_dc=colors_dc,
            opacities=opacities,
            sh_rest=sh_rest,
            is_compressed=True
        )

    def _compute_chunk_indices_sequential(self, num_splats: int) -> np.ndarray:
        """Compute chunk indices assuming sequential grouping of 256 splats per chunk"""
        if not self.chunk_data:
            return None

        splat_indices = np.arange(num_splats, dtype=np.int64)
        chunk_indices = (splat_indices // self.chunk_data.chunk_size).astype(np.int32)
        np.clip(chunk_indices, 0, self.chunk_data.chunk_count - 1, out=chunk_indices)
        return chunk_indices

    def _unpack_position_with_indices(self, packed_pos: np.ndarray, chunk_indices: np.ndarray) -> np.ndarray:
        """Unpack compressed position data using chunk min/max values"""
        if not self.chunk_data:
            raise RuntimeError("Missing chunk data for position unpacking")

        N = packed_pos.shape[0]
        positions = np.empty((N, 3), dtype=np.float32)

        # Extract quantized values (11-10-11 bit layout)
        xq = ((packed_pos >> 21) & 0x7FF).astype(np.float32) / 2047.0
        yq = ((packed_pos >> 11) & 0x3FF).astype(np.float32) / 1023.0
        zq = ((packed_pos >> 0) & 0x7FF).astype(np.float32) / 2047.0

        # Dequantize using chunk-specific min/max values
        unique_chunks = np.unique(chunk_indices)
        for chunk_idx in unique_chunks:
            mask = (chunk_indices == chunk_idx)
            c = int(np.clip(chunk_idx, 0, self.chunk_data.chunk_count - 1))

            min_x, max_x = self.chunk_data.min_x[c], self.chunk_data.max_x[c]
            min_y, max_y = self.chunk_data.min_y[c], self.chunk_data.max_y[c]
            min_z, max_z = self.chunk_data.min_z[c], self.chunk_data.max_z[c]

            positions[mask, 0] = min_x + xq[mask] * (max_x - min_x)
            positions[mask, 1] = min_y + yq[mask] * (max_y - min_y)
            positions[mask, 2] = min_z + zq[mask] * (max_z - min_z)

        return positions

    def _unpack_scale_with_indices(self, packed_scale: np.ndarray, chunk_indices: np.ndarray) -> np.ndarray:
        """Unpack compressed scale data (in log space)"""
        if not self.chunk_data:
            raise RuntimeError("Missing chunk data for scale unpacking")

        N = packed_scale.shape[0]
        scales = np.empty((N, 3), dtype=np.float32)

        # Extract quantized values (11-10-11 bit layout)
        sxq = ((packed_scale >> 21) & 0x7FF).astype(np.float32) / 2047.0
        syq = ((packed_scale >> 11) & 0x3FF).astype(np.float32) / 1023.0
        szq = ((packed_scale >> 0) & 0x7FF).astype(np.float32) / 2047.0

        # Dequantize using chunk-specific min/max values (in log domain)
        unique_chunks = np.unique(chunk_indices)
        for chunk_idx in unique_chunks:
            mask = (chunk_indices == chunk_idx)
            c = int(np.clip(chunk_idx, 0, self.chunk_data.chunk_count - 1))

            min_sx, max_sx = self.chunk_data.min_scale_x[c], self.chunk_data.max_scale_x[c]
            min_sy, max_sy = self.chunk_data.min_scale_y[c], self.chunk_data.max_scale_y[c]
            min_sz, max_sz = self.chunk_data.min_scale_z[c], self.chunk_data.max_scale_z[c]

            # Interpolate in log space, then keep in log space (don't exp here)
            scales[mask, 0] = min_sx + sxq[mask] * (max_sx - min_sx)
            scales[mask, 1] = min_sy + syq[mask] * (max_sy - min_sy)
            scales[mask, 2] = min_sz + szq[mask] * (max_sz - min_sz)

        return scales

    def _unpack_rotation_supersplat(self, packed_rot: np.ndarray) -> np.ndarray:
        """Unpack SuperSplat compressed rotation data"""
        pr = packed_rot.astype(np.uint32)
        # largest_idx = pr & 0x3

        # Extract the three stored components (each 10 bits)
        # SuperSplat packs with left shifts, so first component is in highest bits
        largest_idx = (pr >> 30) & 0x3        # ✅ 从最高2位取
        comp_bits   = pr & 0x3FFFFFFF         # ✅ 低30位是3个10bit分量
        c2 = (comp_bits >> 20) & 0x3FF        # 最高10位 = 第一个被 pack 的分量
        c1 = (comp_bits >> 10) & 0x3FF
        c0 = (comp_bits >>  0) & 0x3FF        # 最低10位 = 最后一个被 pack 的分量        


        def unpack(val):
            n = val.astype(np.float32) / 1023.0            # [0,1]
            return (n * 2.0 - 1.0) * (1.0 / np.sqrt(2.0))  # [-1/√2, +1/√2]

        v0, v1, v2 = unpack(c0), unpack(c1), unpack(c2)    # 注意：v2 对应最高10bit=最先写入

        N = pr.shape[0]
        q = np.zeros((N, 4), dtype=np.float32)  # [w, x, y, z]

        # Reconstruct quaternion based on SuperSplat's component ordering
        # Components are stored in order [x,y,z,w], skipping the largest component
        # First non-largest component -> c2 (highest bits), second -> c1, third -> c0
        for largest in range(4):
            mask = (largest_idx == largest)
            if not np.any(mask):
                continue

            # Build list of non-largest components in SuperSplat's [x,y,z,w] order
            stored_components = [i for i in range(4) if i != largest]

            # Map XYZW indices to our WXYZ quaternion format
            # SuperSplat: [x=0, y=1, z=2, w=3] -> Our format: [w=0, x=1, y=2, z=3]
            supersplat_to_ours = [3, 1, 2, 0]  # [w, x, y, z]

            # Assign extracted values: c2 -> first stored, c1 -> second stored, c0 -> third stored
            our_idx_0 = supersplat_to_ours[stored_components[0]]
            our_idx_1 = supersplat_to_ours[stored_components[1]]
            our_idx_2 = supersplat_to_ours[stored_components[2]]

            q[mask, our_idx_0] = v2[mask]  # First non-largest component (highest bits)
            q[mask, our_idx_1] = v1[mask]  # Second non-largest component (middle bits)
            q[mask, our_idx_2] = v0[mask]  # Third non-largest component (lowest bits)

            # Calculate the largest component (always non-negative)
            s = 1.0 - (q[mask, our_idx_0]**2 + q[mask, our_idx_1]**2 + q[mask, our_idx_2]**2)
            our_largest_idx = supersplat_to_ours[largest]
            q[mask, our_largest_idx] = np.sqrt(np.clip(s, 0.0, 1.0))

        # Normalize quaternions
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(norm, 1e-8)

        return q

    def _unpack_color(self, packed_color: np.ndarray,
                      chunk_indices: Optional[np.ndarray] = None,
                      order: str = 'RGBA',
                      return_logit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack compressed color and opacity data"""
        val = packed_color.astype(np.uint32).reshape(-1)
        shifts = {'RGBA': (24,16,8,0), 'BGRA': (8,16,24,0), 'ARGB': (16,8,0,24), 'ABGR': (0,8,16,24)}[order]
        r8 = ((val >> shifts[0]) & 0xFF).astype(np.float32)
        g8 = ((val >> shifts[1]) & 0xFF).astype(np.float32)
        b8 = ((val >> shifts[2]) & 0xFF).astype(np.float32)
        a8 = ((val >> shifts[3]) & 0xFF).astype(np.float32)
        alpha = (a8 / 255.0).reshape(-1, 1).astype(np.float32)  # Opacity for rendering 0..1

        use_chunk_color = (self.chunk_data is not None and hasattr(self.chunk_data, 'min_r'))
        if use_chunk_color:
            if chunk_indices is None:
                chunk_indices = self._compute_chunk_indices_sequential(len(val))
            ci = np.clip(chunk_indices, 0, self.chunk_data.chunk_count - 1)

            min_r = self.chunk_data.min_r[ci].astype(np.float32)
            min_g = self.chunk_data.min_g[ci].astype(np.float32)
            min_b = self.chunk_data.min_b[ci].astype(np.float32)
            max_r = self.chunk_data.max_r[ci].astype(np.float32)
            max_g = self.chunk_data.max_g[ci].astype(np.float32)
            max_b = self.chunk_data.max_b[ci].astype(np.float32)

            t_r = r8 / 255.0; t_g = g8 / 255.0; t_b = b8 / 255.0
            dr = (max_r - min_r); dr[np.abs(dr) < 1e-12] = 0.0
            dg = (max_g - min_g); dg[np.abs(dg) < 1e-12] = 0.0
            db = (max_b - min_b); db[np.abs(db) < 1e-12] = 0.0

            f_dc_r = min_r + t_r * dr
            f_dc_g = min_g + t_g * dg
            f_dc_b = min_b + t_b * db
            colors_dc = np.stack([f_dc_r, f_dc_g, f_dc_b], axis=1).astype(np.float32)
            colors_dc = (colors_dc - 0.5) / SH_C0
        else:
            # Fallback: old format pack8888 display colors -> DC
            display = np.column_stack([r8, g8, b8]) / 255.0
            colors_dc = ((display - 0.5) / SH_C0).astype(np.float32)

        if return_logit:
            eps = 1e-9
            a = np.clip(alpha, eps, 1.0 - eps)
            op = -np.log(1.0 / a - 1.0)  # Convert to logit space
            return colors_dc, op.astype(np.float32)
        else:
            return colors_dc, alpha

    def _unpack_sh_coefficients(self, sh_element, chunk_indices: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Read quantized uint8 f_rest_* from 'element sh' and dequantize to float SH coefficients.
        SuperSplat quantization rules (see serializePlyCompressed):
          n = sh / 8 + 0.5
          q = clamp(trunc(n * 256), 0, 255)
        Therefore unbiased dequantization using bin centers:
          n_hat = (q + 0.5) / 256
          sh = 8 * (n_hat - 0.5) = (q - 127.5) / 32
        Storage order is channel-major: [R(0..K-1), G(0..K-1), B(0..K-1)].
        This function returns flattened (N, 3*K) float array maintaining that order;
        later in GaussianSplatData.get_sh_coefficients it's reshaped to (N, K, 3) and combined with DC.
        """
        prop_names = []
        if hasattr(sh_element, 'properties'):
            prop_names = [p.name for p in sh_element.properties]

        # Count actual f_rest_i columns (consecutive starting from 0)
        cols = []
        i = 0
        while f'f_rest_{i}' in prop_names:
            cols.append(f'f_rest_{i}')
            i += 1

        if not cols:
            print("No f_rest_* properties found in 'sh' element; skipping SH loading.")
            return None

        # Read as uint8, then dequantize to float using bin centers
        # Result matrix shape (N, 3*K), order maintained as [R_block | G_block | B_block]
        arr_u8 = [np.asarray(sh_element[name], dtype=np.uint8) for name in cols]
        sh_u8 = np.column_stack(arr_u8)

        # Dequantize: sh = (q - 127.5) / 32
        sh = ((sh_u8 - 127.5) / 32.0).astype(np.float32)


        return sh




def main():
    """Example usage and testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Load and analyze Gaussian Splat PLY files")
    parser.add_argument('ply_file', help='Path to PLY file')
    parser.add_argument('--print-sample', action='store_true', help='Print sample data points')
    parser.add_argument('--max-samples', type=int, default=5, help='Maximum number of samples to print')

    args = parser.parse_args()

    # Load the PLY file
    try:
        loader = PLYGaussianLoader()
        gaussian_data = loader.load_ply(args.ply_file)
        gaussian_data.print_info()

        if args.print_sample and gaussian_data.num_splats > 0:
            n_samples = min(args.max_samples, gaussian_data.num_splats)
            print(f"\nFirst {n_samples} sample splats:")

            for i in range(n_samples):
                print(f"Splat {i}:")
                print(f"  Position: [{gaussian_data.positions[i, 0]:.6f}, "
                      f"{gaussian_data.positions[i, 1]:.6f}, {gaussian_data.positions[i, 2]:.6f}]")
                print(f"  Rotation: [{gaussian_data.rotations[i, 0]:.6f}, "
                      f"{gaussian_data.rotations[i, 1]:.6f}, {gaussian_data.rotations[i, 2]:.6f}, "
                      f"{gaussian_data.rotations[i, 3]:.6f}]")
                print(f"  Scale: [{gaussian_data.scales[i, 0]:.6f}, "
                      f"{gaussian_data.scales[i, 1]:.6f}, {gaussian_data.scales[i, 2]:.6f}]")
                print(f"  Color DC: [{gaussian_data.colors_dc[i, 0]:.6f}, "
                      f"{gaussian_data.colors_dc[i, 1]:.6f}, {gaussian_data.colors_dc[i, 2]:.6f}]")
                print(f"  Opacity: {gaussian_data.opacities[i, 0]:.6f}")
                print()

    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())