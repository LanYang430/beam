"""
Feature extraction for molecular dynamics trajectories.

Supports:
- xyz: Cartesian coordinates (default)
- distance: Pairwise distances
- custom: User-defined feature function
"""

import numpy as np

class FeatureExtractor:
    """
    Extract features from trajectory positions.
    
    Parameters
    ----------
    feature_type : str
        Type of features: 'xyz', 'distance', or 'custom'
    custom_function : callable, optional
        Function for custom features. Must have signature:
        custom_function(positions) -> array of shape (n_frames, n_features)
        where positions has shape (n_frames, n_atoms, 3)
    distance_pairs : list of tuples, optional
        For feature_type='distance', specify atom pairs as [(i,j), ...]
        If None, computes all pairwise distances
        
    Examples
    --------
    >>> # XYZ coordinates (default)
    >>> extractor = FeatureExtractor(feature_type='xyz')
    >>> features = extractor.transform(positions)
    
    >>> # Pairwise distances
    >>> extractor = FeatureExtractor(feature_type='distance')
    >>> features = extractor.transform(positions)
    
    >>> # Custom CV
    >>> def rg(positions):
    ...     # Radius of gyration
    ...     mean_pos = positions.mean(axis=1, keepdims=True)
    ...     rg = np.sqrt(np.mean(np.sum((positions - mean_pos)**2, axis=2), axis=1))
    ...     return rg.reshape(-1, 1)
    >>> extractor = FeatureExtractor(feature_type='custom', custom_function=rg)
    >>> features = extractor.transform(positions)
    """
    
    def __init__(self, feature_type='xyz', custom_function=None, distance_pairs=None):
        valid_types = ['xyz', 'distance', 'custom']
        if feature_type not in valid_types:
            raise ValueError(
                f"feature_type must be one of {valid_types}, got '{feature_type}'"
            )
        
        self.feature_type = feature_type
        self.custom_function = custom_function
        self.distance_pairs = distance_pairs
        
        # Validate
        if feature_type == 'custom' and custom_function is None:
            raise ValueError("custom_function must be provided when feature_type='custom'")
    
    def transform(self, positions):
        """
        Extract features from positions.
        
        Parameters
        ----------
        positions : np.ndarray
            Trajectory positions, shape (n_frames, n_atoms, 3)
            
        Returns
        -------
        features : np.ndarray
            Feature matrix, shape (n_frames, n_features)
        """
        if positions.ndim != 3:
            raise ValueError(
                f"positions must be 3D array (n_frames, n_atoms, 3), "
                f"got shape {positions.shape}"
            )
        
        if self.feature_type == 'xyz':
            return self._extract_xyz(positions)
        elif self.feature_type == 'distance':
            return self._extract_distances(positions)
        elif self.feature_type == 'custom':
            return self._extract_custom(positions)
    
    def _extract_xyz(self, positions):
        """Flatten XYZ coordinates"""
        n_frames = positions.shape[0]
        return positions.reshape(n_frames, -1)
    
    def _extract_distances(self, positions):
        """
        Compute pairwise distances.
        
        If distance_pairs is specified, only compute those pairs.
        Otherwise, compute all pairwise distances (can be large!).
        """
        n_frames, n_atoms, _ = positions.shape
        
        if self.distance_pairs is not None:
            # Only specified pairs
            n_pairs = len(self.distance_pairs)
            distances = np.zeros((n_frames, n_pairs))
            
            for frame_idx in range(n_frames):
                for pair_idx, (i, j) in enumerate(self.distance_pairs):
                    vec = positions[frame_idx, i] - positions[frame_idx, j]
                    distances[frame_idx, pair_idx] = np.linalg.norm(vec)
            
            return distances
        else:
            # All pairwise distances
            # Warning: this is O(N^2) atoms!
            if n_atoms > 100:
                import warnings
                warnings.warn(
                    f"Computing all pairwise distances for {n_atoms} atoms. "
                    f"This will create {n_atoms*(n_atoms-1)//2} features. "
                    f"Consider specifying distance_pairs."
                )
            
            n_pairs = n_atoms * (n_atoms - 1) // 2
            distances = np.zeros((n_frames, n_pairs))
            
            for frame_idx in range(n_frames):
                pair_idx = 0
                for i in range(n_atoms):
                    for j in range(i+1, n_atoms):
                        vec = positions[frame_idx, i] - positions[frame_idx, j]
                        distances[frame_idx, pair_idx] = np.linalg.norm(vec)
                        pair_idx += 1
            
            return distances
    
    def _extract_custom(self, positions):
        """Call user-provided function"""
        try:
            features = self.custom_function(positions)
        except Exception as e:
            raise RuntimeError(
                f"Error in custom_function: {e}\n"
                f"custom_function must accept positions (n_frames, n_atoms, 3) "
                f"and return features (n_frames, n_features)"
            )
        
        # Validate output
        if not isinstance(features, np.ndarray):
            raise TypeError(
                f"custom_function must return np.ndarray, got {type(features)}"
            )
        
        if features.ndim != 2:
            raise ValueError(
                f"custom_function must return 2D array (n_frames, n_features), "
                f"got shape {features.shape}"
            )
        
        if features.shape[0] != positions.shape[0]:
            raise ValueError(
                f"custom_function returned wrong number of frames: "
                f"expected {positions.shape[0]}, got {features.shape[0]}"
            )
        
        return features


# Convenience function
def extract_features(positions, feature_type='xyz', **kwargs):
    """
    Convenience function for feature extraction.
    
    Parameters
    ----------
    positions : np.ndarray
        Trajectory positions (n_frames, n_atoms, 3)
    feature_type : str
        'xyz', 'distance', or 'custom'
    **kwargs
        Passed to FeatureExtractor
        
    Returns
    -------
    features : np.ndarray
        Feature matrix (n_frames, n_features)
        
    Examples
    --------
    >>> features = extract_features(positions, feature_type='xyz')
    >>> features = extract_features(positions, feature_type='distance', 
    ...                            distance_pairs=[(0,10), (5,15)])
    >>> features = extract_features(positions, feature_type='custom',
    ...                            custom_function=my_cv_func)
    """
    extractor = FeatureExtractor(feature_type=feature_type, **kwargs)
    return extractor.transform(positions)
