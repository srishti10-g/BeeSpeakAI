import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.audio_processing import BeeAudioProcessor
from src.feature_extraction import BeeFeatureExtractor
from src.clustering_model import BeeClusterAnalyzer
from src.visualization import BeeHealthVisualizer
import json

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if hasattr(obj, 'dtype'):
            if np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            elif np.issubdtype(obj.dtype, np.bool_):
                return bool(obj)
            elif np.issubdtype(obj.dtype, np.complexfloating):
                return {'real': obj.real, 'imag': obj.imag}
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_synthetic_bee_sounds(num_samples=100):
    """Generate synthetic bee sounds for demonstration"""
    print(f"🔊 Generating {num_samples} synthetic bee sounds for demonstration...")
    
    all_features = []
    
    # Simulate different hive conditions through feature patterns
    conditions = {
        'healthy': {'freq_mean': 220, 'freq_std': 0.2, 'hnr': 2.5, 'activity': 0.6},
        'stressed': {'freq_mean': 280, 'freq_std': 0.5, 'hnr': 1.2, 'activity': 0.9},
        'queenless': {'freq_mean': 200, 'freq_std': 0.8, 'hnr': 0.8, 'activity': 0.3},
        'temperature_stress': {'freq_mean': 320, 'freq_std': 0.3, 'hnr': 1.8, 'activity': 0.7},
        'swarming': {'freq_mean': 260, 'freq_std': 0.1, 'hnr': 2.0, 'activity': 0.95}
    }
    
    samples_per_condition = max(1, num_samples // len(conditions))
    
    for condition, params in conditions.items():
        for i in range(samples_per_condition):
            # Create synthetic feature set based on condition parameters
            features = {
                'fundamental_freq_mean': float(np.random.normal(params['freq_mean'], 10)),
                'fundamental_freq_std': float(np.random.normal(params['freq_std'], 0.1)),
                'harmonic_noise_ratio': float(np.random.normal(params['hnr'], 0.3)),
                'activity_ratio': float(np.random.normal(params['activity'], 0.1)),
                'spectral_centroid_mean': float(np.random.normal(250, 50)),
                'spectral_centroid_std': float(np.random.normal(50, 10)),
                'rms_energy': float(np.random.normal(0.1, 0.02)),
                'rms_energy_std': float(np.random.normal(0.02, 0.005)),
                'zero_crossing_rate': float(np.random.normal(0.1, 0.02))
            }
            
            # Add some MFCC-like features
            for j in range(1, 6):
                features[f'mfcc_{j}_mean'] = float(np.random.normal(0, 1))
                features[f'mfcc_{j}_std'] = float(np.random.normal(0.5, 0.1))
            
            all_features.append(features)
    
    return all_features

def clean_feature_data(feature_data):
    """Clean feature data by removing NaN values and ensuring all values are finite"""
    cleaned_data = []
    
    for features in feature_data:
        cleaned_features = {}
        for key, value in features.items():
            # Handle NaN, infinity, and non-finite values
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    # Replace with reasonable defaults based on feature type
                    if 'freq' in key:
                        cleaned_features[key] = 220.0  # Default bee frequency
                    elif 'ratio' in key or 'rate' in key:
                        cleaned_features[key] = 0.5  # Default ratio
                    elif 'energy' in key:
                        cleaned_features[key] = 0.1  # Default energy
                    elif 'mean' in key:
                        cleaned_features[key] = 0.0  # Default mean
                    elif 'std' in key:
                        cleaned_features[key] = 1.0  # Default std
                    else:
                        cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = value
            else:
                cleaned_features[key] = value
        cleaned_data.append(cleaned_features)
    
    return cleaned_data

def analyze_real_audio_files(audio_directory):
    """Analyze real audio files from directory"""
    print(f"🎵 Analyzing real audio files from {audio_directory}...")
    
    processor = BeeAudioProcessor()
    extractor = BeeFeatureExtractor()
    
    all_features = []
    
    # Check if directory exists
    if not os.path.exists(audio_directory):
        print(f"❌ Directory {audio_directory} not found.")
        return []
    
    # Get all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    for file in os.listdir(audio_directory):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(audio_directory, file))
    
    if not audio_files:
        print("❌ No audio files found.")
        return []
    
    print(f"✅ Found {len(audio_files)} audio files")
    
    # Process each audio file
    for i, audio_file in enumerate(audio_files):
        try:
            print(f"   Processing {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
            
            # Load and preprocess audio
            audio, sr = processor.load_audio(audio_file)
            processed_audio = processor.preprocess_audio(audio)
            
            # Extract features
            features = extractor.extract_all_features(processed_audio)
            features['audio_file'] = os.path.basename(audio_file)
            
            all_features.append(features)
            
        except Exception as e:
            print(f"   ❌ Error processing {audio_file}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(all_features)} audio files")
    
    # Clean the feature data
    cleaned_features = clean_feature_data(all_features)
    print(f"🔧 Cleaned {len(all_features) - len(cleaned_features)} problematic feature entries")
    
    return cleaned_features

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'dtype'):
        # Handle numpy types
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # Try to convert to native type
        try:
            return float(obj) if isinstance(obj, (np.number)) else str(obj)
        except:
            return str(obj)

def analyze_single_audio_sample(feature_data):
    """Analyze a single audio sample and compare it with synthetic patterns"""
    print("🔍 Analyzing single audio sample against known patterns...")
    
    # Generate synthetic patterns for comparison
    synthetic_data = generate_synthetic_bee_sounds(50)
    all_features = feature_data + synthetic_data
    
    # Clean all feature data
    all_features_cleaned = clean_feature_data(all_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features_cleaned)
    
    # Remove any remaining NaN values
    clustering_df = features_df.select_dtypes(include=[np.number])
    clustering_df = clustering_df.fillna(0)  # Replace NaN with 0
    
    print(f"✅ Loaded {len(features_df)} total samples (1 real + {len(synthetic_data)} synthetic)")
    
    # Perform analysis
    analyzer = BeeClusterAnalyzer()
    features_scaled = analyzer.prepare_features(clustering_df)
    
    # Check for any remaining NaN values
    if np.isnan(features_scaled).any():
        print("⚠️  NaN values detected in scaled features, replacing with 0")
        features_scaled = np.nan_to_num(features_scaled, nan=0.0)
    
    # Use fixed number of clusters for small datasets
    optimal_clusters = min(3, len(features_scaled) - 1)
    print(f"🎯 Using {optimal_clusters} clusters for analysis")
    
    try:
        cluster_labels = analyzer.cluster_analysis(features_scaled, n_clusters=optimal_clusters)
        cluster_profiles = analyzer.interpret_clusters(features_scaled, cluster_labels)
        
        # Identify which cluster contains our real sample
        real_sample_cluster = cluster_labels[0]  # First sample is our real one
        
        print(f"🎯 Your audio sample belongs to: Cluster {real_sample_cluster}")
        print(f"📊 Inferred status: {cluster_profiles.get(real_sample_cluster, {}).get('inferred_status', 'Unknown')}")
        
        return features_scaled, cluster_labels, cluster_profiles, real_sample_cluster, analyzer
        
    except Exception as e:
        print(f"❌ Cluster analysis failed: {e}")
        print("🔄 Using fallback analysis...")
        # Create simple fallback results
        cluster_labels = np.zeros(len(features_scaled), dtype=int)
        cluster_profiles = {0: {'inferred_status': 'Analysis Failed - Check Audio Quality'}}
        return features_scaled, cluster_labels, cluster_profiles, 0, analyzer

def create_simple_visualization(feature_data, cluster_profiles, real_cluster):
    """Create a simple visualization when dimensionality reduction fails"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    status = cluster_profiles.get(real_cluster, {}).get('inferred_status', 'Unknown')
    
    # Create a simple bar chart showing key features if available
    if feature_data and len(feature_data) > 0:
        real_features = feature_data[0]
        key_features = {k: v for k, v in real_features.items() 
                       if any(x in k for x in ['freq', 'ratio', 'energy', 'activity']) 
                       and isinstance(v, (int, float))}
        
        if key_features:
            # Take top 5 features
            top_features = dict(list(key_features.items())[:5])
            features_names = list(top_features.keys())
            features_values = list(top_features.values())
            
            bars = ax.barh(features_names, features_values, color='lightblue')
            ax.set_xlabel('Feature Value')
            ax.set_title(f'Bee Audio Analysis - Status: {status}')
            
            # Add value labels on bars
            for bar, value in zip(bars, features_values):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2f}', ha='left', va='center')
        else:
            ax.text(0.5, 0.5, f'Bee Health Status: {status}\n\nAudio analysis completed!\n\nKey features extracted successfully.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, f'Bee Health Status: {status}\n\nAudio analysis completed!', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    print("🐝 HiveMind AI - Bee Health Analysis System")
    print("=" * 50)
    
    # Initialize variables to avoid scope issues
    features_scaled = None
    cluster_labels = None
    cluster_profiles = None
    real_cluster = None
    analyzer = None
    feature_data = None
    clustering_df = None
    
    # Step 1: Choose data source
    print("\n📁 Choose data source:")
    print("1. Use synthetic data (demo)")
    print("2. Use real audio files from 'data/raw_audio/'")
    print("3. Analyze single real audio with pattern matching")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    try:
        if choice == "2":
            feature_data = analyze_real_audio_files('data/raw_audio')
            if len(feature_data) < 5:
                print(f"\n⚠️  Only {len(feature_data)} audio files found.")
                print("🔧 Switching to single sample analysis mode...")
                if len(feature_data) > 0:
                    features_scaled, cluster_labels, cluster_profiles, real_cluster, analyzer = analyze_single_audio_sample(feature_data)
                else:
                    print("❌ No valid audio files to analyze. Using synthetic data.")
                    feature_data = generate_synthetic_bee_sounds()
                    features_df = pd.DataFrame(feature_data)
                    clustering_df = features_df.select_dtypes(include=[np.number])
                    analyzer = BeeClusterAnalyzer()
                    features_scaled = analyzer.prepare_features(clustering_df)
                    cluster_labels = analyzer.cluster_analysis(features_scaled, n_clusters=3)
                    cluster_profiles = analyzer.interpret_clusters(features_scaled, cluster_labels)
            else:
                # Normal analysis with sufficient real data
                features_df = pd.DataFrame(feature_data)
                clustering_df = features_df.select_dtypes(include=[np.number])
                clustering_df = clustering_df.fillna(0)  # Clean NaN values
                print(f"✅ Loaded {len(features_df)} audio samples with {len(clustering_df.columns)} features")
                
                analyzer = BeeClusterAnalyzer()
                features_scaled = analyzer.prepare_features(clustering_df)
                optimal_clusters, _, _ = analyzer.find_optimal_clusters(features_scaled)
                print(f"🎯 Optimal number of clusters: {optimal_clusters}")
                cluster_labels = analyzer.cluster_analysis(features_scaled, n_clusters=optimal_clusters)
                cluster_profiles = analyzer.interpret_clusters(features_scaled, cluster_labels)
                
        elif choice == "3":
            # Single sample analysis mode
            feature_data = analyze_real_audio_files('data/raw_audio')
            if len(feature_data) == 0:
                print("❌ No audio files found for analysis.")
                return
            features_scaled, cluster_labels, cluster_profiles, real_cluster, analyzer = analyze_single_audio_sample(feature_data)
            
        else:  # choice 1 or default
            feature_data = generate_synthetic_bee_sounds()
            features_df = pd.DataFrame(feature_data)
            clustering_df = features_df.select_dtypes(include=[np.number])
            print(f"✅ Loaded {len(features_df)} audio samples with {len(clustering_df.columns)} features")
            
            analyzer = BeeClusterAnalyzer()
            features_scaled = analyzer.prepare_features(clustering_df)
            optimal_clusters, _, _ = analyzer.find_optimal_clusters(features_scaled)
            print(f"🎯 Optimal number of clusters: {optimal_clusters}")
            cluster_labels = analyzer.cluster_analysis(features_scaled, n_clusters=optimal_clusters)
            cluster_profiles = analyzer.interpret_clusters(features_scaled, cluster_labels)
        
        # Step 4: Visualize results
        print("📊 Creating visualizations...")
        visualizer = BeeHealthVisualizer()
        
        # Handle visualization creation
        try:
            if features_scaled is not None and len(features_scaled) > 1:
                reduced_features = analyzer.reduce_dimensionality(features_scaled)
                fig = visualizer.plot_cluster_analysis(reduced_features, cluster_labels, cluster_profiles)
            else:
                # Use simple visualization for single samples or when reduction fails
                fig = create_simple_visualization(feature_data, cluster_profiles, real_cluster if real_cluster is not None else 0)
        except Exception as e:
            print(f"⚠️  Advanced visualization failed: {e}")
            print("🔄 Creating simple visualization...")
            fig = create_simple_visualization(feature_data, cluster_profiles, real_cluster if real_cluster is not None else 0)
        
        plt.savefig('bee_health_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: bee_health_analysis.png")
        
        # Create interactive dashboard (skip if it fails)
        try:
            if features_scaled is not None:
                clustering_df_for_viz = pd.DataFrame(features_scaled)
                interactive_fig = visualizer.create_interactive_dashboard(clustering_df_for_viz, cluster_labels, cluster_profiles)
                interactive_fig.write_html('bee_health_dashboard.html')
                print("✅ Saved: bee_health_dashboard.html")
        except Exception as e:
            print(f"⚠️  Could not create interactive dashboard: {e}")
        
        # Step 5: Save results
        print("💾 Saving analysis results...")
        
        # Get audio files list safely
        audio_files_list = []
        if feature_data and any('audio_file' in features for features in feature_data):
            audio_files_list = [features.get('audio_file', f'sample_{i}.wav') for i, features in enumerate(feature_data)]
        else:
            audio_files_list = [f"sample_{i+1}.wav" for i in range(len(feature_data) if feature_data else 0)]
        
        # Convert all numpy types to native Python types
        if cluster_labels is not None:
            try:
                cluster_labels_python = cluster_labels.tolist()
            except:
                cluster_labels_python = [int(x) for x in cluster_labels]
        else:
            cluster_labels_python = []
        
        # Ensure cluster_profiles are serializable
        cluster_profiles_serializable = {}
        if cluster_profiles:
            for cluster_id, profile in cluster_profiles.items():
                try:
                    cluster_profiles_serializable[int(cluster_id)] = convert_numpy_types(profile)
                except Exception as e:
                    print(f"⚠️  Warning: Could not serialize cluster {cluster_id}: {e}")
                    cluster_profiles_serializable[int(cluster_id)] = {
                        'inferred_status': profile.get('inferred_status', 'Unknown'),
                        'error': 'Full profile not serializable'
                    }
        
        results = {
            'cluster_labels': cluster_labels_python,
            'cluster_profiles': cluster_profiles_serializable,
            'feature_names': analyzer.feature_names if analyzer and hasattr(analyzer, 'feature_names') else [],
            'audio_files': audio_files_list,
            'analysis_summary': {
                'total_samples': int(len(feature_data) if feature_data else 0),
                'optimal_clusters': int(len(cluster_profiles) if cluster_profiles else 1),
                'features_used': int(len(clustering_df.columns) if clustering_df is not None else 0)
            }
        }
        
        # Save with error handling
        try:
            with open('cluster_results.json', 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            print("✅ Saved: cluster_results.json")
        except Exception as e:
            print(f"⚠️  Could not save JSON results: {e}")
        
        # Step 6: Display results
        print("\n" + "=" * 50)
        print("🎯 CLUSTER ANALYSIS RESULTS")
        print("=" * 50)
        
        if choice == "3" and real_cluster is not None:
            # Single sample analysis results
            real_status = cluster_profiles.get(real_cluster, {}).get('inferred_status', 'Unknown')
            print(f"\n🎯 YOUR BEE HIVE ANALYSIS:")
            print(f"📊 Health Status: {real_status}")
            print(f"🔍 Pattern Group: Cluster {real_cluster}")
            
            # Show characteristics of the matched cluster
            profile = cluster_profiles.get(real_cluster, {})
            if 'fundamental_freq_mean' in profile:
                freq = profile['fundamental_freq_mean']['mean']
                print(f"📏 Avg Frequency: {freq:.1f} Hz")
            if 'harmonic_noise_ratio' in profile:
                hnr = profile['harmonic_noise_ratio']['mean']
                print(f"🎵 Harmony Score: {hnr:.2f}")
            if 'activity_ratio' in profile:
                activity = profile['activity_ratio']['mean']
                print(f"🔊 Activity Level: {activity:.2f}")
        else:
            # Multiple sample analysis results
            if cluster_profiles:
                for cluster_id, profile in cluster_profiles.items():
                    status = profile.get('inferred_status', 'Unknown')
                    if cluster_labels is not None:
                        cluster_count = np.sum(cluster_labels == cluster_id)
                        percentage = (cluster_count / len(cluster_labels)) * 100
                    else:
                        cluster_count = 0
                        percentage = 0
                    
                    print(f"\n📊 Cluster {cluster_id}: {status}")
                    print(f"   Samples: {cluster_count} ({percentage:.1f}%)")
                    
                    # Show key characteristics
                    if 'fundamental_freq_mean' in profile:
                        freq = profile['fundamental_freq_mean']['mean']
                        print(f"   Avg Frequency: {freq:.1f} Hz")
                    if 'harmonic_noise_ratio' in profile:
                        hnr = profile['harmonic_noise_ratio']['mean']
                        print(f"   Harmony Score: {hnr:.2f}")
                    if 'activity_ratio' in profile:
                        activity = profile['activity_ratio']['mean']
                        print(f"   Activity Level: {activity:.2f}")
        
        print("\n" + "=" * 50)
        print("✅ Analysis complete!")
        print("📁 Generated files:")
        print("   - bee_health_analysis.png (Static visualization)")
        print("   - bee_health_dashboard.html (Interactive dashboard)")
        print("   - cluster_results.json (Detailed results)")
        
        print("\n🐝 Next steps:")
        print("   • Add more bee recordings to 'data/raw_audio/' for better analysis")
        print("   • The system will learn your hive's unique acoustic patterns!")
        print("   • Compare recordings from different days to track hive health changes")
        
        # Show the plot
        try:
            plt.show()
        except:
            print("📊 Visualizations saved but cannot display plot in this environment")
            
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        print("💡 Tips for better results:")
        print("   • Ensure your audio file is clear and has bee sounds")
        print("   • Try recording in a quieter environment")
        print("   • Make sure the audio file is not corrupted")
        print("   • Try option 1 first to test the system with synthetic data")

if __name__ == "__main__":
    main()
