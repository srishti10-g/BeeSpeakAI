import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np

class BeeHealthVisualizer:
    def __init__(self):
        plt.style.use('default')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plot_cluster_analysis(self, reduced_features, cluster_labels, cluster_profiles):
        """Create comprehensive cluster visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Cluster scatter plot
            if reduced_features is not None and len(reduced_features) > 1:
                scatter = axes[0, 0].scatter(reduced_features[:, 0], reduced_features[:, 1], 
                                            c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
                axes[0, 0].set_title('Bee Sound Clusters - Pattern Discovery')
                axes[0, 0].set_xlabel('Component 1')
                axes[0, 0].set_ylabel('Component 2')
                plt.colorbar(scatter, ax=axes[0, 0])
            else:
                axes[0, 0].text(0.5, 0.5, 'Cluster Visualization\n(Not enough data for 2D plot)', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Bee Sound Clusters')
            
            # Plot 2: Cluster distribution
            unique, counts = np.unique(cluster_labels, return_counts=True)
            axes[0, 1].bar(unique, counts, color=self.colors[:len(unique)])
            axes[0, 1].set_title('Cluster Distribution')
            axes[0, 1].set_xlabel('Cluster ID')
            axes[0, 1].set_ylabel('Number of Samples')
            
            # Plot 3: Health status distribution
            if cluster_profiles:
                statuses = []
                counts_by_status = []
                
                for cluster_id in unique:
                    status = cluster_profiles.get(int(cluster_id), {}).get('inferred_status', 'Unknown')
                    statuses.append(status)
                    counts_by_status.append(counts[cluster_id == unique][0])
                
                axes[1, 0].barh(range(len(statuses)), counts_by_status, color=self.colors[:len(statuses)])
                axes[1, 0].set_yticks(range(len(statuses)))
                axes[1, 0].set_yticklabels(statuses, fontsize=9)
                axes[1, 0].set_title('Inferred Health Status Distribution')
                axes[1, 0].set_xlabel('Number of Samples')
            else:
                axes[1, 0].text(0.5, 0.5, 'No cluster profiles available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Health Status Distribution')
            
            # Plot 4: Info panel
            axes[1, 1].text(0.5, 0.7, '🐝 HiveMind AI Analysis\n\nPattern Discovery Complete!\n\n'
                           f'📊 Clusters Found: {len(unique)}\n'
                           f'🎯 Samples Analyzed: {len(cluster_labels)}\n'
                           '🔍 Ready for real bee data!', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 1].set_title('Analysis Summary')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Visualization error: {e}")
            # Return empty figure as fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Visualization Failed\nBut analysis completed!', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def create_interactive_dashboard(self, features_df, cluster_labels, cluster_profiles):
        """Create interactive Plotly dashboard"""
        try:
            from src.clustering_model import BeeClusterAnalyzer
            
            analyzer = BeeClusterAnalyzer()
            features_scaled = analyzer.prepare_features(features_df)
            reduced_features = analyzer.reduce_dimensionality(features_scaled, method='pca')
            
            # Create interactive scatter plot
            plot_df = pd.DataFrame({
                'x': reduced_features[:, 0],
                'y': reduced_features[:, 1],
                'cluster': cluster_labels,
                'status': [cluster_profiles.get(int(label), {}).get('inferred_status', 'Unknown') 
                          for label in cluster_labels]
            })
            
            fig = px.scatter(plot_df, x='x', y='y', color='status',
                            title='🐝 Bee Health Cluster Analysis - Interactive Dashboard',
                            hover_data=['cluster'],
                            color_discrete_sequence=px.colors.qualitative.Set3)
            
            fig.update_layout(
                xaxis_title='Pattern Component 1',
                yaxis_title='Pattern Component 2',
                showlegend=True
            )
            
            return fig
        except Exception as e:
            print(f"Interactive dashboard failed: {e}")
            # Return simple figure
            fig = px.scatter(x=[0], y=[0], title='Dashboard Generation Failed')
            return fig

    def create_simple_health_chart(self, cluster_profiles):
        """Create a simple health status chart for Streamlit"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        statuses = []
        counts = []
        colors = []
        
        for cluster_id, profile in cluster_profiles.items():
            status = profile.get('inferred_status', 'Unknown')
            statuses.append(status)
            counts.append(1)  # Each cluster counts as 1
            
            # Assign colors based on status
            if 'healthy' in status.lower():
                colors.append('#4CAF50')  # Green
            elif 'warning' in status.lower() or 'risk' in status.lower():
                colors.append('#FF9800')  # Orange
            else:
                colors.append('#F44336')  # Red
        
        if statuses:
            bars = ax.bar(statuses, counts, color=colors)
            ax.set_ylabel('Number of Clusters')
            ax.set_title('Bee Health Status Distribution')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

    def create_feature_radar_chart(self, cluster_profiles, feature_names):
        """Create a radar chart for cluster feature comparison"""
        try:
            # Select key features for radar chart
            key_features = ['fundamental_freq_mean', 'harmonic_noise_ratio', 
                           'activity_ratio', 'spectral_centroid_mean', 'rms_energy']
            
            available_features = [f for f in key_features if any(f in name for name in feature_names)]
            
            if len(available_features) < 3:
                return self._create_fallback_chart("Not enough features for radar chart")
            
            # Prepare data for radar chart
            categories = available_features
            fig = plt.figure(figsize=(10, 8))
            
            # Normalize values for radar chart
            for cluster_id, profile in cluster_profiles.items():
                values = []
                for feature in available_features:
                    # Find the feature in profile
                    feature_value = 0
                    for key in profile.keys():
                        if feature in key and 'mean' in key:
                            feature_value = profile[key].get('mean', 0) if isinstance(profile[key], dict) else profile[key]
                            break
                    values.append(feature_value)
                
                # Normalize values to 0-1 scale for radar chart
                if max(values) > 0:
                    values = [v / max(values) for v in values]
                
                # Close the radar chart
                values.append(values[0])
                angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
                angles += angles[:1]
                
                ax = fig.add_subplot(111, polar=True)
                ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title('Feature Comparison Across Clusters')
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            return fig
            
        except Exception as e:
            print(f"Radar chart error: {e}")
            return self._create_fallback_chart("Radar chart generation failed")

    def _create_fallback_chart(self, message):
        """Create a simple fallback chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
        return fig

def create_audio_waveform_plot(audio_data, sample_rate=22050):
    """Create a simple waveform plot of audio data"""
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Create time array
        times = np.arange(len(audio_data)) / sample_rate
        
        ax.plot(times, audio_data, color='#1f77b4', alpha=0.7)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Bee Hive Audio Waveform')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Waveform plot error: {e}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'Waveform visualization failed', 
               ha='center', va='center', transform=ax.transAxes)
        return fig

def create_spectrogram_plot(audio_data, sample_rate=22050):
    """Create a spectrogram plot of audio data"""
    try:
        import librosa.display
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create spectrogram
        stft = librosa.stft(audio_data)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        img = librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', 
                                      y_axis='hz', ax=ax, cmap='viridis')
        
        ax.set_title('Bee Hive Audio Spectrogram')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        
        # Add colorbar
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Spectrogram plot error: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Spectrogram visualization failed', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
