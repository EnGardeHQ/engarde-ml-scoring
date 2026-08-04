"""
Advanced Segmentation Algorithms

This module provides sophisticated segmentation algorithms including
graph-based clustering, ensemble methods, and time-series segmentation.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
# UMAP is optional - imported conditionally below
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from collections import defaultdict, Counter

# Optional ML libraries - import lazily to allow service to load without them
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None
    logger.warning("hdbscan not available - HDBSCAN clustering will be disabled")

try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
    KneeLocator = None
    logger.warning("kneed not available - Elbow detection will be disabled")

try:
    from sklearn.manifold import TSNE
    # UMAP is imported from sklearn.decomposition above, but may not be available
    try:
        from umap import UMAP as UMAPImport
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        UMAPImport = None
        logger.warning("umap-learn not available - UMAP dimensionality reduction will be disabled")
except ImportError:
    UMAP_AVAILABLE = False
    UMAPImport = None

logger = logging.getLogger(__name__)


class SegmentationAlgorithm(str, Enum):
    """Available segmentation algorithms"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan" 
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    SPECTRAL = "spectral"
    HDBSCAN = "hdbscan"
    GRAPH_BASED = "graph_based"
    ENSEMBLE = "ensemble"
    TIME_SERIES = "time_series"
    BEHAVIORAL_COHORT = "behavioral_cohort"
    VALUE_BASED = "value_based"


class DimensionalityReduction(str, Enum):
    """Dimensionality reduction techniques"""
    PCA = "pca"
    UMAP = "umap"
    TSNE = "tsne"
    NONE = "none"


@dataclass
class SegmentationConfig:
    """Configuration for segmentation algorithms"""
    algorithm: SegmentationAlgorithm
    n_clusters: Optional[int] = None
    auto_determine_clusters: bool = True
    min_cluster_size: int = 50
    max_clusters: int = 20
    dimensionality_reduction: DimensionalityReduction = DimensionalityReduction.NONE
    target_dimensions: int = 10
    
    # Algorithm-specific parameters
    kmeans_params: Dict[str, Any] = None
    dbscan_params: Dict[str, Any] = None
    hierarchical_params: Dict[str, Any] = None
    spectral_params: Dict[str, Any] = None
    
    # Feature selection
    feature_selection_method: str = "variance"
    feature_importance_threshold: float = 0.01
    
    # Validation
    cross_validation_folds: int = 5
    stability_iterations: int = 10


@dataclass
class ClusterValidationMetrics:
    """Cluster validation metrics"""
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    adjusted_rand_score: Optional[float] = None
    stability_score: float = 0.0
    inertia: Optional[float] = None
    n_clusters: int = 0
    n_noise_points: int = 0


@dataclass
class SegmentationResult:
    """Result of advanced segmentation"""
    algorithm_used: str
    cluster_labels: np.ndarray
    cluster_centers: Optional[np.ndarray] = None
    validation_metrics: ClusterValidationMetrics = None
    feature_importance: Dict[str, float] = None
    reduced_features: Optional[np.ndarray] = None
    execution_time: float = 0.0


class AdvancedSegmentationEngine:
    """Engine for advanced segmentation algorithms"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.dimensionality_reducer = None
        self.trained_models = {}
        
    def segment_audience(
        self,
        features: np.ndarray,
        config: SegmentationConfig,
        feature_names: Optional[List[str]] = None
    ) -> SegmentationResult:
        """
        Perform advanced audience segmentation
        
        Args:
            features: Feature matrix
            config: Segmentation configuration
            feature_names: Names of features
            
        Returns:
            SegmentationResult with detailed results
        """
        start_time = datetime.utcnow()
        
        # Preprocess features
        processed_features = self._preprocess_features(features, config)
        
        # Determine optimal number of clusters if needed
        if config.auto_determine_clusters and config.n_clusters is None:
            config.n_clusters = self._determine_optimal_clusters(
                processed_features, config
            )
        
        # Apply segmentation algorithm
        result = self._apply_algorithm(processed_features, config)
        
        # Calculate validation metrics
        result.validation_metrics = self._calculate_validation_metrics(
            processed_features, result.cluster_labels, config
        )
        
        # Calculate feature importance
        if feature_names:
            result.feature_importance = self._calculate_feature_importance(
                processed_features, result.cluster_labels, feature_names
            )
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        result.execution_time = execution_time
        
        logger.info(
            f"Segmentation completed using {config.algorithm.value}: "
            f"{result.validation_metrics.n_clusters} clusters, "
            f"silhouette score: {result.validation_metrics.silhouette_score:.3f}, "
            f"execution time: {execution_time:.2f}s"
        )
        
        return result
    
    def ensemble_segmentation(
        self,
        features: np.ndarray,
        algorithms: List[SegmentationAlgorithm],
        feature_names: Optional[List[str]] = None
    ) -> SegmentationResult:
        """
        Perform ensemble segmentation using multiple algorithms
        
        Args:
            features: Feature matrix
            algorithms: List of algorithms to ensemble
            feature_names: Names of features
            
        Returns:
            Combined segmentation result
        """
        logger.info(f"Starting ensemble segmentation with {len(algorithms)} algorithms")
        
        # Run individual algorithms
        individual_results = []
        for algorithm in algorithms:
            config = SegmentationConfig(algorithm=algorithm)
            try:
                result = self.segment_audience(features, config, feature_names)
                individual_results.append(result)
            except Exception as e:
                logger.warning(f"Algorithm {algorithm.value} failed: {str(e)}")
                continue
        
        if not individual_results:
            raise ValueError("All algorithms failed")
        
        # Combine results using consensus clustering
        consensus_labels = self._consensus_clustering(
            [result.cluster_labels for result in individual_results]
        )
        
        # Create ensemble result
        ensemble_result = SegmentationResult(
            algorithm_used="ensemble",
            cluster_labels=consensus_labels,
            validation_metrics=self._calculate_validation_metrics(
                features, consensus_labels, SegmentationConfig(SegmentationAlgorithm.ENSEMBLE)
            )
        )
        
        logger.info(f"Ensemble segmentation completed with {len(np.unique(consensus_labels))} clusters")
        
        return ensemble_result
    
    def time_series_segmentation(
        self,
        time_series_data: Dict[str, np.ndarray],
        window_size: int = 30,
        overlap: float = 0.5
    ) -> SegmentationResult:
        """
        Perform time-series based segmentation
        
        Args:
            time_series_data: Dictionary of user_id -> time series values
            window_size: Size of time windows
            overlap: Overlap between windows (0-1)
            
        Returns:
            Time-series segmentation result
        """
        logger.info(f"Starting time-series segmentation with window size {window_size}")
        
        # Extract features from time series
        features = self._extract_time_series_features(
            time_series_data, window_size, overlap
        )
        
        # Apply clustering to time-series features
        config = SegmentationConfig(
            algorithm=SegmentationAlgorithm.KMEANS,
            auto_determine_clusters=True
        )
        
        result = self.segment_audience(features, config)
        result.algorithm_used = "time_series"
        
        return result
    
    def behavioral_cohort_analysis(
        self,
        user_actions: Dict[str, List[Dict[str, Any]]],
        cohort_period: str = "monthly"
    ) -> SegmentationResult:
        """
        Perform behavioral cohort analysis
        
        Args:
            user_actions: Dictionary of user_id -> list of actions
            cohort_period: Cohort period (daily, weekly, monthly)
            
        Returns:
            Cohort-based segmentation result
        """
        logger.info(f"Starting behavioral cohort analysis with {cohort_period} periods")
        
        # Create cohort features
        cohort_features = self._create_cohort_features(user_actions, cohort_period)
        
        # Apply clustering to cohort features
        config = SegmentationConfig(
            algorithm=SegmentationAlgorithm.HIERARCHICAL,
            auto_determine_clusters=True
        )
        
        result = self.segment_audience(cohort_features, config)
        result.algorithm_used = "behavioral_cohort"
        
        return result
    
    def value_based_segmentation(
        self,
        user_values: Dict[str, Dict[str, float]],
        value_metrics: List[str] = None
    ) -> SegmentationResult:
        """
        Perform value-based segmentation (RFM-style)
        
        Args:
            user_values: Dictionary of user_id -> value metrics
            value_metrics: List of value metrics to use
            
        Returns:
            Value-based segmentation result
        """
        if value_metrics is None:
            value_metrics = ['recency', 'frequency', 'monetary']
        
        logger.info(f"Starting value-based segmentation using metrics: {value_metrics}")
        
        # Create value feature matrix
        features = self._create_value_features(user_values, value_metrics)
        
        # Apply specialized value-based clustering
        result = self._value_based_clustering(features, value_metrics)
        
        return result
    
    def graph_based_segmentation(
        self,
        similarity_matrix: np.ndarray,
        edge_threshold: float = 0.5
    ) -> SegmentationResult:
        """
        Perform graph-based segmentation using community detection
        
        Args:
            similarity_matrix: User similarity matrix
            edge_threshold: Threshold for creating edges
            
        Returns:
            Graph-based segmentation result
        """
        logger.info("Starting graph-based segmentation")
        
        # Create graph from similarity matrix
        graph = self._create_similarity_graph(similarity_matrix, edge_threshold)
        
        # Detect communities
        communities = self._detect_communities(graph)
        
        # Convert communities to cluster labels
        cluster_labels = self._communities_to_labels(communities, similarity_matrix.shape[0])
        
        result = SegmentationResult(
            algorithm_used="graph_based",
            cluster_labels=cluster_labels,
            validation_metrics=self._calculate_validation_metrics(
                similarity_matrix, cluster_labels, 
                SegmentationConfig(SegmentationAlgorithm.GRAPH_BASED)
            )
        )
        
        return result
    
    # Private methods
    
    def _preprocess_features(
        self,
        features: np.ndarray,
        config: SegmentationConfig
    ) -> np.ndarray:
        """Preprocess features before clustering"""
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Feature selection
        if config.feature_selection_method == "variance":
            selected_features = self._variance_feature_selection(
                scaled_features, config.feature_importance_threshold
            )
        else:
            selected_features = scaled_features
        
        # Dimensionality reduction
        if config.dimensionality_reduction != DimensionalityReduction.NONE:
            reduced_features = self._apply_dimensionality_reduction(
                selected_features, config
            )
            return reduced_features
        
        return selected_features
    
    def _variance_feature_selection(
        self,
        features: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Select features based on variance threshold"""
        variances = np.var(features, axis=0)
        selected_indices = variances > threshold
        
        if np.sum(selected_indices) == 0:
            # Keep all features if none meet threshold
            return features
        
        return features[:, selected_indices]
    
    def _apply_dimensionality_reduction(
        self,
        features: np.ndarray,
        config: SegmentationConfig
    ) -> np.ndarray:
        """Apply dimensionality reduction"""
        target_dim = min(config.target_dimensions, features.shape[1])
        
        if config.dimensionality_reduction == DimensionalityReduction.PCA:
            reducer = PCA(n_components=target_dim, random_state=42)
        elif config.dimensionality_reduction == DimensionalityReduction.UMAP:
            if UMAP_AVAILABLE and UMAPImport:
                try:
                    reducer = UMAPImport(n_components=target_dim, random_state=42)
                except Exception as e:
                    logger.warning(f"UMAP failed: {e}, falling back to PCA")
                    reducer = PCA(n_components=target_dim, random_state=42)
            else:
                logger.warning("UMAP not available, falling back to PCA")
                reducer = PCA(n_components=target_dim, random_state=42)
        elif config.dimensionality_reduction == DimensionalityReduction.TSNE:
            reducer = TSNE(n_components=min(target_dim, 3), random_state=42)
        else:
            return features
        
        self.dimensionality_reducer = reducer
        reduced_features = reducer.fit_transform(features)
        
        return reduced_features
    
    def _determine_optimal_clusters(
        self,
        features: np.ndarray,
        config: SegmentationConfig
    ) -> int:
        """Determine optimal number of clusters using multiple methods"""
        max_clusters = min(config.max_clusters, len(features) // config.min_cluster_size)
        max_clusters = max(2, max_clusters)
        
        # Method 1: Elbow method with KMeans
        elbow_k = self._elbow_method(features, max_clusters)
        
        # Method 2: Silhouette analysis
        silhouette_k = self._silhouette_analysis(features, max_clusters)
        
        # Method 3: Gap statistic (simplified)
        gap_k = self._gap_statistic(features, max_clusters)
        
        # Combine methods using voting
        candidates = [elbow_k, silhouette_k, gap_k]
        candidates = [k for k in candidates if k is not None]
        
        if not candidates:
            return min(5, max_clusters)
        
        # Use median as final choice
        optimal_k = int(np.median(candidates))
        return max(2, min(optimal_k, max_clusters))
    
    def _elbow_method(self, features: np.ndarray, max_clusters: int) -> Optional[int]:
        """Find optimal clusters using elbow method"""
        try:
            inertias = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
            
            # Use KneeLocator to find elbow (if available)
            if KNEED_AVAILABLE and KneeLocator:
                kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
                return kl.elbow if kl.elbow else None
            else:
                # Fallback: use simple heuristic (find point with max curvature)
                logger.warning("KneeLocator not available, using simple heuristic")
                # Simple heuristic: find point with maximum curvature
                if len(inertias) < 3:
                    return None
                # Calculate second derivative (curvature)
                first_diff = np.diff(inertias)
                second_diff = np.diff(first_diff)
                # Find maximum curvature point
                max_curvature_idx = np.argmax(np.abs(second_diff)) + 1
                return k_range[max_curvature_idx] if max_curvature_idx < len(k_range) else None
            
        except Exception as e:
            logger.warning(f"Elbow method failed: {str(e)}")
            return None
    
    def _silhouette_analysis(self, features: np.ndarray, max_clusters: int) -> Optional[int]:
        """Find optimal clusters using silhouette analysis"""
        try:
            silhouette_scores = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                score = silhouette_score(features, labels)
                silhouette_scores.append(score)
            
            best_k_idx = np.argmax(silhouette_scores)
            return k_range[best_k_idx]
            
        except Exception as e:
            logger.warning(f"Silhouette analysis failed: {str(e)}")
            return None
    
    def _gap_statistic(self, features: np.ndarray, max_clusters: int) -> Optional[int]:
        """Find optimal clusters using gap statistic (simplified)"""
        try:
            def compute_inertia(X, labels):
                centers = np.array([X[labels == i].mean(axis=0) for i in np.unique(labels)])
                inertia = sum([np.sum((X[labels == i] - centers[i]) ** 2) for i in np.unique(labels)])
                return inertia
            
            gaps = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                # Real data clustering
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                real_inertia = compute_inertia(features, labels)
                
                # Reference data (uniform random)
                ref_inertias = []
                for _ in range(5):  # Reduced iterations for speed
                    ref_data = np.random.uniform(
                        features.min(axis=0), features.max(axis=0), features.shape
                    )
                    ref_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    ref_labels = ref_kmeans.fit_predict(ref_data)
                    ref_inertia = compute_inertia(ref_data, ref_labels)
                    ref_inertias.append(ref_inertia)
                
                gap = np.log(np.mean(ref_inertias)) - np.log(real_inertia)
                gaps.append(gap)
            
            # Find first local maximum
            for i in range(1, len(gaps)):
                if gaps[i] < gaps[i-1]:
                    return k_range[i-1]
            
            return k_range[np.argmax(gaps)]
            
        except Exception as e:
            logger.warning(f"Gap statistic failed: {str(e)}")
            return None
    
    def _apply_algorithm(
        self,
        features: np.ndarray,
        config: SegmentationConfig
    ) -> SegmentationResult:
        """Apply specific clustering algorithm"""
        
        if config.algorithm == SegmentationAlgorithm.KMEANS:
            return self._apply_kmeans(features, config)
        elif config.algorithm == SegmentationAlgorithm.DBSCAN:
            return self._apply_dbscan(features, config)
        elif config.algorithm == SegmentationAlgorithm.HIERARCHICAL:
            return self._apply_hierarchical(features, config)
        elif config.algorithm == SegmentationAlgorithm.GAUSSIAN_MIXTURE:
            return self._apply_gaussian_mixture(features, config)
        elif config.algorithm == SegmentationAlgorithm.SPECTRAL:
            return self._apply_spectral(features, config)
        elif config.algorithm == SegmentationAlgorithm.HDBSCAN:
            return self._apply_hdbscan(features, config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    
    def _apply_kmeans(self, features: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
        """Apply KMeans clustering"""
        params = config.kmeans_params or {}
        default_params = {'n_init': 10, 'random_state': 42}
        default_params.update(params)
        
        model = KMeans(n_clusters=config.n_clusters, **default_params)
        labels = model.fit_predict(features)
        
        return SegmentationResult(
            algorithm_used="kmeans",
            cluster_labels=labels,
            cluster_centers=model.cluster_centers_
        )
    
    def _apply_dbscan(self, features: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
        """Apply DBSCAN clustering"""
        params = config.dbscan_params or {}
        default_params = {'eps': 0.5, 'min_samples': 5}
        default_params.update(params)
        
        model = DBSCAN(**default_params)
        labels = model.fit_predict(features)
        
        return SegmentationResult(
            algorithm_used="dbscan",
            cluster_labels=labels
        )
    
    def _apply_hierarchical(self, features: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
        """Apply hierarchical clustering"""
        params = config.hierarchical_params or {}
        default_params = {'linkage': 'ward'}
        default_params.update(params)
        
        model = AgglomerativeClustering(n_clusters=config.n_clusters, **default_params)
        labels = model.fit_predict(features)
        
        return SegmentationResult(
            algorithm_used="hierarchical",
            cluster_labels=labels
        )
    
    def _apply_gaussian_mixture(self, features: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
        """Apply Gaussian Mixture Model"""
        model = GaussianMixture(n_components=config.n_clusters, random_state=42)
        labels = model.fit_predict(features)
        
        return SegmentationResult(
            algorithm_used="gaussian_mixture",
            cluster_labels=labels,
            cluster_centers=model.means_
        )
    
    def _apply_spectral(self, features: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
        """Apply spectral clustering"""
        params = config.spectral_params or {}
        default_params = {'random_state': 42}
        default_params.update(params)
        
        model = SpectralClustering(n_clusters=config.n_clusters, **default_params)
        labels = model.fit_predict(features)
        
        return SegmentationResult(
            algorithm_used="spectral",
            cluster_labels=labels
        )
    
    def _apply_hdbscan(self, features: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
        """Apply HDBSCAN clustering"""
        try:
            if not HDBSCAN_AVAILABLE or hdbscan is None:
                raise ValueError("HDBSCAN is not installed. Install with: pip install hdbscan")
            model = hdbscan.HDBSCAN(min_cluster_size=config.min_cluster_size)
            labels = model.fit_predict(features)
            
            return SegmentationResult(
                algorithm_used="hdbscan",
                cluster_labels=labels
            )
        except ImportError:
            logger.warning("HDBSCAN not available, falling back to DBSCAN")
            return self._apply_dbscan(features, config)
    
    def _calculate_validation_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        config: SegmentationConfig
    ) -> ClusterValidationMetrics:
        """Calculate cluster validation metrics"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_noise = np.sum(labels == -1)
        
        metrics = ClusterValidationMetrics(
            silhouette_score=0.0,
            calinski_harabasz_score=0.0,
            davies_bouldin_score=0.0,
            n_clusters=n_clusters,
            n_noise_points=n_noise
        )
        
        # Only calculate metrics if we have valid clusters
        if n_clusters > 1 and len(np.unique(labels[labels != -1])) > 1:
            try:
                # Filter out noise points for metric calculation
                valid_mask = labels != -1
                if np.sum(valid_mask) > 1:
                    valid_features = features[valid_mask]
                    valid_labels = labels[valid_mask]
                    
                    if len(np.unique(valid_labels)) > 1:
                        metrics.silhouette_score = silhouette_score(valid_features, valid_labels)
                        metrics.calinski_harabasz_score = calinski_harabasz_score(valid_features, valid_labels)
                        
                        from sklearn.metrics import davies_bouldin_score
                        metrics.davies_bouldin_score = davies_bouldin_score(valid_features, valid_labels)
                        
            except Exception as e:
                logger.warning(f"Error calculating validation metrics: {str(e)}")
        
        return metrics
    
    def _calculate_feature_importance(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance for clustering"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Filter out noise points
            valid_mask = labels != -1
            if np.sum(valid_mask) < 10:
                return {}
            
            valid_features = features[valid_mask]
            valid_labels = labels[valid_mask]
            
            # Train random forest to predict cluster labels
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(valid_features, valid_labels)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(importance)
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def _consensus_clustering(self, label_sets: List[np.ndarray]) -> np.ndarray:
        """Combine multiple clustering results using consensus"""
        n_samples = len(label_sets[0])
        
        # Create co-association matrix
        co_matrix = np.zeros((n_samples, n_samples))
        
        for labels in label_sets:
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j] and labels[i] != -1:
                        co_matrix[i, j] += 1
        
        # Normalize by number of algorithms
        co_matrix /= len(label_sets)
        
        # Apply hierarchical clustering to co-association matrix
        condensed_matrix = squareform(1 - co_matrix)  # Convert similarity to distance
        linkage_matrix = linkage(condensed_matrix, method='average')
        
        # Determine number of clusters using gap in dendrogram
        n_clusters = self._determine_consensus_clusters(linkage_matrix)
        consensus_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        return consensus_labels - 1  # Convert to 0-based indexing
    
    def _determine_consensus_clusters(self, linkage_matrix: np.ndarray) -> int:
        """Determine optimal number of clusters for consensus"""
        # Simple heuristic: look for largest gap in linkage distances
        distances = linkage_matrix[:, 2]
        
        if len(distances) < 2:
            return 2
        
        # Find largest increase in distance
        diff = np.diff(distances)
        if len(diff) == 0:
            return 2
        
        largest_gap_idx = np.argmax(diff)
        n_clusters = len(distances) - largest_gap_idx
        
        return max(2, min(n_clusters, 10))  # Reasonable bounds
    
    def _extract_time_series_features(
        self,
        time_series_data: Dict[str, np.ndarray],
        window_size: int,
        overlap: float
    ) -> np.ndarray:
        """Extract features from time series data"""
        features = []
        
        for user_id, series in time_series_data.items():
            user_features = []
            
            # Sliding window features
            step_size = int(window_size * (1 - overlap))
            for start in range(0, len(series) - window_size + 1, step_size):
                window = series[start:start + window_size]
                
                # Statistical features
                user_features.extend([
                    np.mean(window),
                    np.std(window),
                    np.min(window),
                    np.max(window),
                    np.median(window),
                    np.percentile(window, 25),
                    np.percentile(window, 75)
                ])
            
            # Trend features
            if len(series) > 1:
                trend = np.polyfit(range(len(series)), series, 1)[0]
                user_features.append(trend)
            else:
                user_features.append(0.0)
            
            # Seasonality features (simplified)
            if len(series) >= 7:  # Weekly seasonality
                weekly_pattern = np.mean(series.reshape(-1, 7), axis=0)
                user_features.extend(weekly_pattern)
            
            features.append(user_features)
        
        return np.array(features)
    
    def _create_cohort_features(
        self,
        user_actions: Dict[str, List[Dict[str, Any]]],
        cohort_period: str
    ) -> np.ndarray:
        """Create features for cohort analysis"""
        features = []
        
        for user_id, actions in user_actions.items():
            user_features = []
            
            # Convert actions to time series
            action_counts = self._actions_to_time_series(actions, cohort_period)
            
            # Cohort features
            if len(action_counts) > 0:
                user_features.extend([
                    len(action_counts),  # Number of active periods
                    np.sum(action_counts),  # Total actions
                    np.mean(action_counts),  # Average actions per period
                    np.max(action_counts),  # Peak activity
                    np.sum(action_counts > 0) / len(action_counts)  # Activity rate
                ])
            else:
                user_features.extend([0, 0, 0, 0, 0])
            
            # Retention features
            if len(action_counts) >= 2:
                retention = np.sum(action_counts[1:] > 0) / (len(action_counts) - 1)
                user_features.append(retention)
            else:
                user_features.append(0.0)
            
            features.append(user_features)
        
        return np.array(features)
    
    def _actions_to_time_series(
        self,
        actions: List[Dict[str, Any]],
        period: str
    ) -> np.ndarray:
        """Convert user actions to time series"""
        # Simplified implementation
        # In practice, this would properly handle date parsing and aggregation
        
        if not actions:
            return np.array([])
        
        # Group actions by period and count
        period_counts = defaultdict(int)
        for action in actions:
            # This would use actual date parsing in practice
            period_key = action.get('date', '2024-01-01')[:7]  # YYYY-MM
            period_counts[period_key] += 1
        
        return np.array(list(period_counts.values()))
    
    def _create_value_features(
        self,
        user_values: Dict[str, Dict[str, float]],
        value_metrics: List[str]
    ) -> np.ndarray:
        """Create feature matrix from value metrics"""
        features = []
        
        for user_id, values in user_values.items():
            user_features = []
            for metric in value_metrics:
                user_features.append(values.get(metric, 0.0))
            features.append(user_features)
        
        return np.array(features)
    
    def _value_based_clustering(
        self,
        features: np.ndarray,
        value_metrics: List[str]
    ) -> SegmentationResult:
        """Perform specialized value-based clustering"""
        # Use quantile-based segmentation for RFM-style analysis
        n_quantiles = 5  # Quintiles
        
        # Calculate quantiles for each metric
        quantiles = []
        for i in range(features.shape[1]):
            metric_quantiles = np.quantile(features[:, i], np.linspace(0, 1, n_quantiles + 1))
            quantiles.append(metric_quantiles)
        
        # Assign quintile labels
        quintile_labels = np.zeros_like(features, dtype=int)
        for i in range(features.shape[1]):
            quintile_labels[:, i] = np.digitize(features[:, i], quantiles[i]) - 1
            quintile_labels[:, i] = np.clip(quintile_labels[:, i], 0, n_quantiles - 1)
        
        # Create combined cluster labels
        cluster_labels = []
        for row in quintile_labels:
            # Create unique identifier for each combination
            label = ''.join(map(str, row))
            cluster_labels.append(hash(label) % 20)  # Limit to 20 clusters
        
        cluster_labels = np.array(cluster_labels)
        
        return SegmentationResult(
            algorithm_used="value_based",
            cluster_labels=cluster_labels
        )
    
    def _create_similarity_graph(
        self,
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> nx.Graph:
        """Create graph from similarity matrix"""
        graph = nx.Graph()
        
        n_nodes = similarity_matrix.shape[0]
        graph.add_nodes_from(range(n_nodes))
        
        # Add edges above threshold
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if similarity_matrix[i, j] > threshold:
                    graph.add_edge(i, j, weight=similarity_matrix[i, j])
        
        return graph
    
    def _detect_communities(self, graph: nx.Graph) -> List[List[int]]:
        """Detect communities in graph"""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            
            # Convert partition to list of communities
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            return list(communities.values())
            
        except ImportError:
            # Fallback to simple connected components
            return list(nx.connected_components(graph))
    
    def _communities_to_labels(
        self,
        communities: List[List[int]],
        n_samples: int
    ) -> np.ndarray:
        """Convert community lists to cluster labels"""
        labels = np.full(n_samples, -1)  # Initialize with noise label
        
        for community_id, community in enumerate(communities):
            for node in community:
                if node < n_samples:
                    labels[node] = community_id
        
        return labels

# Force refresh: 2025-12-25 10:06:56
