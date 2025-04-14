# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning tools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure

# Other utilities
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlayerRoleClassifier:
    """
    Player role classification based on positions and metrics

    This class contains the knowledge about:
    - What positions exist in soccer (e.g., Centre-Forward, Left-Back)
    - What roles each position can play (e.g., a Centre-Forward can be a Finisher, Target, or Roamer)
    - What statistics are most important for each role (e.g., a Finisher needs goals, shots on target)
    """

    def __init__(self):
        # This dictionary maps soccer positions to possible roles
        # For example, a Centre-Forward could be a Finisher, Target, or Roamer depending on playing style
        self.position_role_mapping = {
            'Centre-Forward': ['Finisher', 'Target', 'Roamer'],
            'Second Striker': ['Finisher', 'Target', 'Roamer'],
            'Right Midfield': ['Wide_Threat', 'Unlocker', 'Outlet'],
            'Right Winger': ['Wide_Threat', 'Unlocker', 'Outlet'],
            'Left Midfield': ['Wide_Threat', 'Unlocker', 'Outlet'],
            'Left Winger': ['Wide_Threat', 'Unlocker', 'Outlet'],
            'Attacking Midfield': ['Orchestrator', 'Creator', 'Box_Crasher'],
            'Central Midfield': ['Builder', 'Distributor', 'Box_to_Box'],
            'Defensive Midfield': ['Builder', 'Distributor', 'Box_to_Box'],
            'Left-Back': ['Safety', 'Progressor', 'Overlapper'],
            'Right-Back': ['Safety', 'Progressor', 'Overlapper'],
            'Centre-Back': ['Anchor', 'Spreader', 'Aggressor'],
            'Sweeper': ['Anchor', 'Spreader', 'Aggressor']
        }

        # For each role, this defines which statistics are most important
        # Updated metrics based on available data columns and role descriptions
        self.role_metrics = {
            # Forward roles
            'Finisher': [
                # Core performance metrics for any striker
                'Gls_Standard', 'G_per_Sh_Standard', 'xG_Expected.x', 'G_per_SoT_Standard', 'Sh_Standard', 'SoT_Standard',
                # Role-specific metrics
                'Att Pen_Touches', 'npxG_Expected.x', 'G_minus_xG_Expected', 'Sh_per_90_Standard', 'SoT_percent_Standard'
            ],
            'Target': [
                # Core performance metrics for any striker
                'Gls_Standard', 'Sh_Standard', 'xG_Expected.x',
                # Role-specific metrics
                'Won_Aerial', 'Won_percent_Aerial', 'Fld', 'Att Pen_Touches', 'Mid 3rd_Touches',
                'Touches_Touches', 'Cmp_percent_Total', 'KP'
            ],
            'Roamer': [
                # Core performance metrics for any striker
                'Gls_Standard', 'Sh_Standard', 'xG_Expected.x',
                # Role-specific metrics
                'Ast.x', 'xA_Expected', 'Succ_Take', 'PrgC_Carries', 'CPA_Carries',
                'KP', 'Carries_Carries', 'PrgP_Progression', 'Final_Third_Carries'
            ],

            # Winger/wide midfielder roles
            'Wide_Threat': [
                # Core performance metrics for wide attackers
                'Succ_Take', 'PrgC_Carries', 'Att 3rd_Touches',
                # Role-specific metrics
                'Gls_Standard', 'SoT_Standard', 'xG_Expected.x', 'Att Pen_Touches',
                'CPA_Carries', 'Rec_Receiving', 'SCA_SCA', 'GCA_GCA'
            ],
            'Unlocker': [
                # Core performance metrics for wide attackers
                'Att 3rd_Touches', 'SCA_SCA',
                # Role-specific metrics
                'Ast.x', 'xA_Expected', 'KP', 'PPA', 'TB_Pass', 'Crs_Pass',
                'Final_Third', 'Sw_Pass', 'PrgP_Progression', 'GCA_GCA'
            ],
            'Outlet': [
                # Core performance metrics for wide attackers
                'Att 3rd_Touches', 'PrgC_Carries',
                # Role-specific metrics
                'Rec_Receiving', 'PrgR_Receiving', 'Fld', 'Succ_Take',
                'Mid 3rd_Touches', 'Carries_Carries', 'Fls', 'Def 3rd_Touches'
            ],

            # Attacking midfielder roles
            'Box_Crasher': [
                # Core performance metrics for attacking midfielders
                'Att 3rd_Touches', 'SCA_SCA',
                # Role-specific metrics
                'Att Pen_Touches', 'Gls_Standard', 'npxG_Expected.x', 'SoT_Standard',
                'PrgC_Carries', 'CPA_Carries', 'Succ_Take', 'PrgR_Receiving'
            ],
            'Creator': [
                # Core performance metrics for attacking midfielders
                'SCA_SCA', 'Att 3rd_Touches',
                # Role-specific metrics
                'Ast.x', 'xA_Expected', 'KP', 'PPA', 'TB_Pass', 'Final_Third',
                'PrgP_Progression', 'GCA_GCA', 'Cmp_percent_Total', 'xAG'
            ],
            'Orchestrator': [
                # Core performance metrics for attacking midfielders
                'Touches_Touches', 'Att 3rd_Touches',
                # Role-specific metrics
                'Cmp_percent_Total', 'PrgP_Progression', 'Final_Third', 'Sw_Pass',
                'KP', 'Mid 3rd_Touches', 'Carries_Carries', 'Recov'
            ],

            # Central midfielder roles
            'Box_to_Box': [
                # Core performance metrics for central midfielders
                'Mid 3rd_Touches', 'Tkl_Tackles',
                # Role-specific metrics
                'Tkl+Int', 'Int.x', 'Recov', 'PrgP_Progression',
                'PrgC_Carries', 'Final_Third_Carries', 'Def 3rd_Touches', 'Att 3rd_Touches'
            ],
            'Distributor': [
                # Core performance metrics for central midfielders
                'Mid 3rd_Touches', 'PrgP_Progression',
                # Role-specific metrics
                'Cmp_percent_Total', 'Cmp_percent_Long', 'Att_Long', 'Sw_Pass',
                'PrgP', 'PPA', 'TotDist_Total', 'PrgDist_Total'
            ],
            'Builder': [
                # Core performance metrics for central midfielders
                'Mid 3rd_Touches', 'Def 3rd_Touches',
                # Role-specific metrics
                'Cmp_percent_Total', 'PrgP_Progression', 'Tkl_Tackles', 'Int.x',
                'Pressures', 'Recov', 'Tkl+Int', 'Touches_Touches'
            ],

            # Fullback roles
            'Overlapper': [
                # Core performance metrics for fullbacks
                'Def 3rd_Touches', 'Tkl_Tackles',
                # Role-specific metrics
                'Crs_Pass', 'PrgC_Carries', 'Att 3rd_Touches', 'KP', 'Succ_Take',
                'Final_Third_Carries', 'Att', 'SCA_SCA'
            ],
            'Progressor': [
                # Core performance metrics for fullbacks
                'Def 3rd_Touches', 'Tkl_Tackles',
                # Role-specific metrics
                'PrgP_Progression', 'PrgC_Carries', 'Succ_Take', 'Final_Third_Carries',
                'Sw_Pass', 'PrgDist_Total', 'TotDist_Total', 'Mid 3rd_Touches'
            ],
            'Safety': [
                # Core performance metrics for fullbacks
                'Def 3rd_Touches', 'Tkl_Tackles',
                # Role-specific metrics
                'TklW_Tackles', 'Int.x', 'Clr', 'Won_Aerial', 'Blocks_Blocks',
                'Tkl+Int', 'Def Pen_Touches', 'Pass_Blocks'
            ],

            # Center back roles
            'Aggressor': [
                # Core performance metrics for center backs
                'Def Pen_Touches', 'Def 3rd_Touches',
                # Role-specific metrics
                'Pressures', 'Tkl_Tackles', 'TklW_Tackles', 'Blocks_Blocks', 'Fls',
                'Int.x', 'PrgC_Carries', 'Mid 3rd_Touches'
            ],
            'Spreader': [
                # Core performance metrics for center backs
                'Def Pen_Touches', 'Def 3rd_Touches',
                # Role-specific metrics
                'Sw_Pass', 'Cmp_Long', 'Att_Long', 'Cmp_percent_Long', 'Cmp_percent_Total',
                'PrgP_Progression', 'TotDist_Total', 'PrgDist_Total'
            ],
            'Anchor': [
                # Core performance metrics for center backs
                'Def Pen_Touches', 'Def 3rd_Touches',
                # Role-specific metrics
                'Int.x', 'TklW_Tackles', 'Pressures', 'Clr', 'Blocks_Blocks',
                'Sh_Blocks', 'Won_Aerial', 'Def Pen_Touches'
            ]
        }

    def get_roles_for_position(self, position: str) -> list:
        """
        Get possible roles for a specific position

        For example, if position is 'Centre-Forward', this returns ['Finisher', 'Target', 'Roamer']
        """
        return self.position_role_mapping.get(position, [])

    def get_metrics_for_roles(self, roles: list) -> list:
        """
        Get all metrics related to specified roles

        This collects all the statistics needed to evaluate a player for specific roles
        """
        metrics = set()  # Using a set to avoid duplicate metrics
        for role in roles:
            if role in self.role_metrics:
                metrics.update(self.role_metrics[role])
        return list(metrics)

    def get_position_group(self, position: str) -> list:
        """
        Find positions with similar roles to the given position

        This helps when looking for players who could potentially play in different positions
        """
        if position not in self.position_role_mapping:
            return []

        # Get the roles for this position
        possible_roles = self.position_role_mapping[position]

        # Find all positions that share at least one role with our position
        return [pos for pos, roles in self.position_role_mapping.items()
                if any(role in roles for role in possible_roles)]


class PlayerRecommendationSystem:
    """
    This system recommends similar soccer/football players, analyzes player roles,
    and groups players with similar playing styles using machine learning.

    This class does the heavy lifting:
    - Processes player data
    - Clusters players into role groups
    - Finds similar players
    - Analyzes what role a player is best suited for
    - Creates visualizations of player clusters

    Think of it as a "player finder" that can:
    1. Find players with similar playing styles to a target player
    2. Determine what role a player is best suited for (e.g., Finisher, Creator)
    3. Group players into clusters based on their statistics
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the player recommendation system.

        Parameters:
        - data_path: Optional path to a CSV file containing player statistics

        What happens:
        - Creates a "scaler" to normalize player statistics (makes numbers comparable)
        - Sets up the role classifier (which maps positions to potential roles)
        - Initializes empty variables to store player data
        - If a data file path is provided, loads the data immediately
        """
        # StandardScaler makes all player stats comparable by converting them to a standard scale
        self.scaler = StandardScaler()

        # The classifier contains position-to-role mappings and metrics for each role
        self.classifier = PlayerRoleClassifier()

        # Will hold the original player data (as loaded from CSV)
        self.data = None

        # Will hold the processed/normalized player data (after scaling)
        self.processed_data = None

        # If a data file path is provided, load the data right away
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str) -> None:
        """
        Load player statistics from a CSV file and preprocess it.

        Parameters:
        - data_path: Path to the CSV file

        What happens:
        - Reads the CSV file into a pandas DataFrame (a table-like structure)
        - Calls the preprocess_data method to clean and normalize the data
        - Logs information about successful data loading
        """
        try:
            # Read the CSV file into a pandas DataFrame
            self.data = pd.read_csv(data_path)

            # Process the data (handle missing values, scale numeric features)
            self.processed_data = self.preprocess_data(self.data)

            # Log success message with the number of players loaded
            logger.info(f"Data loaded and processed: {len(self.data)} players")
        except Exception as e:
            # If anything goes wrong, log the error and re-raise it
            logger.error(f"Error loading data: {str(e)}")
            raise

    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set player data directly from a DataFrame (instead of loading from a CSV).

        Parameters:
        - data: A pandas DataFrame containing player data

        What happens:
        - Makes a copy of the provided DataFrame
        - Preprocesses the data
        - Logs information about successful data setting
        """
        # Make a copy of the input data to avoid modifying the original
        self.data = data.copy()

        # Process the data (handle missing values, scale numeric features)
        self.processed_data = self.preprocess_data(self.data)

        # Log success message with the number of players
        logger.info(f"Data set and processed: {len(self.data)} players")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize player statistics data.

        Parameters:
        - data: Raw player data

        Returns:
        - Processed player data with normalized values

        What happens:
        - Creates a copy of the data to avoid modifying the original
        - Identifies all numeric columns (player statistics)
        - Fills in any missing values with average values
        - Scales all numeric features to a standard range
        """
        # Make a copy of the input data to avoid modifying the original
        processed_data = data.copy()

        # Select only numeric columns (player statistics)
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns


        # Fill missing values with the mean (average) for each column
        # This avoids errors when processing the data
        # processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].mean())
        processed_data[numeric_cols] = processed_data[numeric_cols].fillna(0)

        # Create a list of columns to scale, excluding 'age'
        cols_to_scale = [col for col in numeric_cols if col != 'Age']

        # Scale numeric features so they're all on a similar scale
        # This is important for fair comparisons between different stats
        # (e.g., goals and passes have very different ranges)
        processed_data[cols_to_scale] = self.scaler.fit_transform(processed_data[cols_to_scale])

        return processed_data

    def _ensure_data_loaded(self) -> None:
        """
        Check that data has been loaded before performing operations.

        What happens:
        - Raises an error if no data has been loaded yet
        """
        if self.processed_data is None:
            raise ValueError("Data not loaded. Call load_data() or set_data() first")

    def cluster_players(self,
                        position_group: str,
                        n_clusters: int = 3,
                        filter_positions: bool = True,
                        custom_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Group players into clusters based on their playing style.

        Parameters:
        - position_group: Position to analyze (e.g., "Centre-Forward")
        - n_clusters: Number of groups to create
        - filter_positions: Whether to only include players from similar positions
        - custom_metrics: Optional specific statistics to focus on

        Returns:
        - Dictionary containing clustered data and visualization information

        What happens:
        - Checks that data is loaded
        - Filters players by position if requested
        - Identifies relevant metrics (statistics) for clustering
        - Uses KMeans algorithm to group players into clusters
        - Performs dimensionality reduction (PCA) for visualization
        - Analyzes characteristics of each cluster
        - Maps clusters to potential player roles
        """
        # Make sure data is loaded before proceeding
        self._ensure_data_loaded()

        # Create a copy of the data to work with
        data = self.processed_data.copy()

        # Filter by position if requested
        if filter_positions:
            # Get all positions with similar roles to the specified position
            position_groups = self.classifier.get_position_group(position_group)
            if not position_groups:
                raise ValueError(f"Invalid position group: {position_group}")

            # Keep only players in those positions
            data = data[data['TmPos'].isin(position_groups)]

            # Make sure we have at least one player
            if len(data) == 0:
                raise ValueError(f"No players found for position group: {position_group}")

        # Determine which metrics (statistics) to use for clustering
        if custom_metrics:
            # Use the custom metrics if provided
            metrics = custom_metrics
        else:
            # Otherwise, get metrics based on possible roles for the position
            possible_roles = self.classifier.get_roles_for_position(position_group)
            metrics = self.classifier.get_metrics_for_roles(possible_roles)

        # Make sure the metrics exist in our data
        available_metrics = [m for m in metrics if m in data.columns]

        if len(available_metrics) == 0:
            raise ValueError("No valid metrics found for clustering")

        # Extract just the metrics we need for clustering
        X = data[available_metrics]

        # Perform KMeans clustering to group players
        # n_clusters = number of groups to create
        # random_state = for reproducible results
        # n_init = number of times algorithm runs with different starting points
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        data['cluster'] = kmeans.fit_predict(X)

        # Get the center of each cluster (the "average" player in that group)
        cluster_centers = kmeans.cluster_centers_

        # Use PCA to reduce dimensions for visualization
        # This takes all our metrics and projects them onto 2 dimensions for plotting
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Add PCA coordinates to our data for plotting
        data['pca_1'] = X_pca[:, 0]
        data['pca_2'] = X_pca[:, 1]

        # Calculate center coordinates in PCA space
        centers_pca = pca.transform(cluster_centers)

        # Calculate how much variance is explained by our 2 PCA components
        explained_variance = pca.explained_variance_ratio_

        # Analyze statistics for each cluster to understand their characteristics
        cluster_stats = self._analyze_cluster_stats(data, available_metrics, n_clusters)

        # Map clusters to potential roles based on position
        possible_roles = self.classifier.get_roles_for_position(position_group)
        role_mapping = {}
        if possible_roles and len(possible_roles) >= n_clusters:
            role_mapping = {i: possible_roles[i] for i in range(n_clusters)}

        # Return a dictionary with all the results
        return {
            'data': data,                         # Player data with cluster assignments
            'metrics_used': available_metrics,    # Statistics used for clustering
            'cluster_centers': cluster_centers,   # Center of each cluster
            'centers_pca': centers_pca,           # Center coordinates in 2D space
            'explained_variance': explained_variance,  # Quality of PCA representation
            'cluster_stats': cluster_stats,       # Statistics for each cluster
            'role_mapping': role_mapping,         # Mapping of clusters to roles
            'position_group': position_group      # Position analyzed
        }

    def _analyze_cluster_stats(self, data: pd.DataFrame, metrics: List[str], n_clusters: int) -> Dict:
        """
        Analyze statistics for each player cluster.

        Parameters:
        - data: Player data with cluster assignments
        - metrics: List of statistics to analyze
        - n_clusters: Number of clusters

        Returns:
        - Dictionary with statistics for each cluster

        What happens:
        - For each cluster, calculates:
          - Size (number of players)
          - Top players in the cluster
          - Average value for each metric
          - Difference from overall average
          - 90th percentile value
        """
        # Dictionary to store results
        cluster_stats = {}

        # Analyze each cluster
        for cluster in range(n_clusters):
            # Get players in this cluster
            cluster_data = data[data['cluster'] == cluster]

            # Initialize stats for this cluster
            cluster_stats[cluster] = {
                'size': len(cluster_data),                    # Number of players
                'top_players': cluster_data['Player'].head(5).tolist(),  # Top 5 players
                'metrics': {}                                 # Will hold metric stats
            }

            # Calculate statistics for each metric
            for metric in metrics:
                if metric in data.columns:
                    # Calculate average for this cluster
                    cluster_avg = cluster_data[metric].mean()

                    # Calculate overall average
                    overall_avg = data[metric].mean()

                    # Calculate difference (how this cluster differs from average)
                    diff = cluster_avg - overall_avg

                    # Store the statistics
                    cluster_stats[cluster]['metrics'][metric] = {
                        'average': cluster_avg,
                        'difference': diff,
                        'percentile': np.percentile(data[metric], 90)  # 90th percentile value
                    }

        return cluster_stats

    def get_role_probabilities(self,
                              player_name: str,
                              position: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate how well a player fits different roles.

        Parameters:
        - player_name: Name of the player to analyze
        - position: Position to consider (uses player's actual position if None)

        Returns:
        - Dictionary mapping roles to probability scores

        What happens:
        - Finds the player in the dataset
        - Determines their position (or uses provided position)
        - Runs clustering on players in that position
        - Calculates distances to each cluster center
        - Converts distances to probabilities using softmax
        """
        # Make sure data is loaded
        self._ensure_data_loaded()

        # Find the player in our dataset
        player_data = self.processed_data[self.processed_data['Player'] == player_name]

        if player_data.empty:
            raise ValueError(f"Player {player_name} not found in dataset")

        # Get the player's position if not provided
        if position is None:
            position = player_data['TmPos'].iloc[0]

        # Get possible roles for this position
        possible_roles = self.classifier.get_roles_for_position(position)

        if not possible_roles:
            raise ValueError(f"No roles defined for position: {position}")

        # Get metrics for these roles
        metrics = self.classifier.get_metrics_for_roles(possible_roles)
        available_metrics = [m for m in metrics if m in self.processed_data.columns]

        # Run clustering to get centers for each role
        cluster_results = self.cluster_players(
            position_group=position,
            n_clusters=len(possible_roles),
            filter_positions=True
        )

        # Get the player's statistics
        player_features = player_data[available_metrics].values.reshape(1, -1)

        # Calculate distances to each cluster center
        distances = []
        for center in cluster_results['cluster_centers']:
            # Euclidean distance between player and cluster center
            dist = np.linalg.norm(player_features - center)
            distances.append(dist)

        # Convert distances to probabilities using softmax
        # Closer distances = higher probabilities
        exp_distances = np.exp(-np.array(distances))
        probabilities = exp_distances / np.sum(exp_distances)

        # Create mapping of roles to probabilities
        role_probs = dict(zip(possible_roles, probabilities))

        return role_probs



    def find_similar_players(self,
                      target_player: str,
                      n_recommendations: int = 5,
                      use_position_groups: bool = True,
                      exclude_current_team: bool = False,
                      custom_metrics: Optional[List[str]] = None,
                      alternative_positions: bool = False,
                      position_weight: float = 0.0) -> pd.DataFrame:
        """
        Find players with similar playing styles.

        This function identifies players who play similarly to a specified target player.
        It works like a "players like this" feature you'd see on sports websites.

        Parameters:
        - target_player: Name of the player to find similarities for
        - n_recommendations: Number of similar players to return
        - use_position_groups: Whether to only consider players in similar positions
        - exclude_current_team: Whether to exclude players from the same team
        - custom_metrics: Optional specific statistics to focus on
        - alternative_positions: Whether to find players in different positions
        - position_weight: How much to prioritize position when finding alternatives

        Returns:
        - DataFrame of similar players with similarity scores

        What happens:
        - Finds the target player in the dataset
        - Filters potential similar players by position if requested
        - Determines relevant metrics for comparison
        - Calculates similarity using cosine similarity
        - Adjusts for positions if looking for alternative positions
        - Returns top similar players with raw data values
        """
        # Step 1: Make sure data is loaded before we try to use it
        self._ensure_data_loaded()
        
        # Step 2: Make a copy of the processed player data for analysis
        player_data = self.processed_data.copy()

        # Step 3: Find our target player in the dataset
        # This creates a smaller DataFrame with just the target player's data
        target_data = player_data[player_data['Player'] == target_player]

        # Check if we actually found the player
        if target_data.empty:
            raise ValueError(f"Player {target_player} not found in dataset")

        # Step 4: Get the player's position and team
        position = target_data['TmPos'].iloc[0]  # Get first value from position column
        # Only get team if that column exists
        team = target_data['Squad'].iloc[0] if 'Squad' in target_data.columns else None

        # Step 5: Filter for players in similar positions if requested
        if use_position_groups:
            position_groups = self.classifier.get_position_group(position)
            print(f"Position groups for {position}: {position_groups}")  # Debug output
            if not position_groups:
                logger.warning(f"No position groups found for {position}, using all players")
                similar_players = player_data
            else:
                similar_players = player_data[player_data['TmPos'].isin(position_groups)]
        else:
            similar_players = player_data
            position_groups = []  # Not used unless alternative_positions=True

        # Step 6: Exclude players from the same team if requested
        # This is useful for finding transfer targets
        if exclude_current_team and team and 'Squad' in similar_players.columns:
            similar_players = similar_players[similar_players['Squad'] != team]

        # Step 7: Determine which stats to use for comparison
        if custom_metrics:
            # If specific metrics were provided, use those
            available_metrics = [m for m in custom_metrics if m in similar_players.columns]
        else:
            # Otherwise, use metrics based on the player's position
            metrics = set()
            if use_position_groups:
                # Get possible roles for this position and their associated metrics
                possible_roles = self.classifier.get_roles_for_position(position)
                for role in possible_roles:
                    # Add all metrics for each possible role
                    metrics.update(self.classifier.role_metrics[role])
            else:
                # If not filtering by position, use all metrics from all roles
                metrics = set([metric for role_metrics in self.classifier.role_metrics.values()
                            for metric in role_metrics])

            # Make sure we only use metrics that are available in our dataset
            available_metrics = list(metrics.intersection(similar_players.columns))

        # Make sure we have at least some metrics to compare
        if len(available_metrics) == 0:
            raise ValueError("No valid metrics found for similarity calculation")

        logger.info(f"Using {len(available_metrics)} metrics for similarity calculation")

        # Step 8: Calculate similarity using cosine similarity
        # Cosine similarity measures the cosine of the angle between two vectors
        # Higher values (closer to 1) mean more similar players
        similarity_matrix = cosine_similarity(
            similar_players[available_metrics],
            target_data[available_metrics]
        )

        # Add similarity scores to our data
        similar_players['similarity'] = similarity_matrix.flatten()

        # Step 9: Handle alternative positions if requested
        if alternative_positions:
            # How much weight to give to position vs. playing style
            position_factor = position_weight

            # Create a score for position matching:
            # 1.0 = exact same position
            # 0.5 = similar position
            # 0.0 = different position group
            similar_players['position_match'] = similar_players['TmPos'].apply(
                lambda p: 1.0 if p == position else 0.5 if p in position_groups else 0.0
            )

            # Combine metrics similarity with position matching
            # For example, if position_factor is 0.7:
            # - 70% of score comes from position match
            # - 30% comes from statistical similarity
            similar_players['adjusted_similarity'] = (
                similar_players['similarity'] * (1 - position_factor) +
                similar_players['position_match'] * position_factor
            )

            # Use adjusted similarity that accounts for position
            sort_column = 'adjusted_similarity'
        else:
            # If not using alternative positions, just use raw similarity
            sort_column = 'similarity'

        # Step 10: Get the top recommendations
        recommendations = (similar_players
                        .sort_values(sort_column, ascending=False)  # Sort by similarity (highest first)
                        .head(n_recommendations + 1)  # Get one extra to account for target player
                        .query('Player != @target_player'))  # Remove target player from results
                        
        # Step 11: Prepare the raw data for output
        # Create a copy of the raw data
        raw_data = self.data.copy()
        
        # Extract key identifiers from recommendations
        player_identifiers = recommendations[['Player', 'TmPos']]
        
        # Create the final result with selected columns and raw data values
        result_players = []
        
        for _, player_row in player_identifiers.iterrows():
            player_name = player_row['Player']
            player_pos = player_row['TmPos']
            
            # Get the player's raw data from self.data
            raw_player_data = raw_data[(raw_data['Player'] == player_name) & 
                                    (raw_data['TmPos'] == player_pos)]
            
            # Get the similarity score from processed recommendations
            similarity_score = recommendations.loc[
                (recommendations['Player'] == player_name) &
                (recommendations['TmPos'] == player_pos), 'similarity'
            ].values[0]
            
            # Create a player result dictionary
            player_result = {}
            
            # Add standard info first
            player_result['Player'] = player_name
            player_result['TmPos'] = player_pos
            
            # Add Squad and Age if available
            if 'Squad' in raw_player_data.columns:
                player_result['Squad'] = raw_player_data['Squad'].iloc[0]
            
            if 'Age' in raw_player_data.columns:
                player_result['Age'] = raw_player_data['Age'].iloc[0]

            if 'player_foot' in raw_player_data.columns:
                player_result['player_foot'] = raw_player_data['player_foot'].iloc[0]

            if 'player_market_value_euro' in raw_player_data.columns:
                player_result['player_market_value_euro'] = raw_player_data['player_market_value_euro'].iloc[0]

            if 'Url' in raw_player_data.columns:
                player_result['Url'] = raw_player_data['Url'].iloc[0]
            
            # Add similarity score
            player_result['similarity'] = similarity_score
            
            # Add adjusted_similarity if we calculated it
            if alternative_positions and 'adjusted_similarity' in recommendations.columns:
                adj_similarity = recommendations.loc[
                    (recommendations['Player'] == player_name) &
                    (recommendations['TmPos'] == player_pos), 'adjusted_similarity'
                ].values[0]
                player_result['adjusted_similarity'] = adj_similarity
            
            # Add all available metrics from raw data
            for metric in available_metrics:
                if metric in raw_player_data.columns:
                    player_result[metric] = raw_player_data[metric].iloc[0]
            
            result_players.append(player_result)
        
        # Convert to DataFrame
        final_results = pd.DataFrame(result_players)
        
        # Return top N recommendations
        return final_results.head(n_recommendations)



    def generate_cluster_visualization(self, cluster_results: Dict[str, Any],
                                      show_annotations: bool = True,
                                      annotation_size: int = 8,
                                      figsize: Tuple[int, int] = (12, 8)) -> Optional[str]:
        """
        Generate a visual plot showing player clusters based on their stats.

        This function creates a visual representation of player clusters
        It helps to see which players are similar and how they group into different playstyles

        Parameters:
        - cluster_results: Results from the cluster_players method
        - show_annotations: Whether to show player names on the plot
        - annotation_size: Font size for the annotations
        - figsize: Size of the figure (width, height)

        Returns:
        - Base64 encoded image string that can be displayed in a web page

        What happens:
        - Creates a 2D scatter plot where each point is a player
        - Colors players based on their assigned cluster/role
        - Marks cluster centers with X
        - Optionally labels players and clusters
        - Returns the plot as an image string
        """
        try:
            # Step 1: Create a new figure with the specified size
            plt.figure(figsize=figsize)

            # Step 2: Extract the data we need from the cluster results
            data = cluster_results['data']  # Player data with cluster assignments
            centers_pca = cluster_results['centers_pca']  # Center coordinates for each cluster
            position_group = cluster_results['position_group']  # Position being analyzed
            role_mapping = cluster_results['role_mapping']  # Which roles correspond to clusters

            # Step 3: Create a custom color palette for the clusters
            # This helps visually distinguish different clusters
            unique_clusters = data['cluster'].unique()
            colors = sns.color_palette("viridis", len(unique_clusters))

            # Step 4: Create the scatter plot
            # Each point represents a player, colored by their cluster
            scatter = plt.scatter(
                data['pca_1'], data['pca_2'],  # X and Y coordinates (from PCA)
                c=data['cluster'],  # Color by cluster
                cmap='viridis',  # Color palette
                s=100,  # Size of points
                alpha=0.8  # Transparency
            )

            # Step 5: Add player names as annotations if requested
            if show_annotations:
                # To avoid cluttering the plot, only show names for top players in each cluster
                for i, cluster in enumerate(unique_clusters):
                    # Get data for just this cluster
                    cluster_data = data[data['cluster'] == cluster]
                    # Get the top 3 players by similarity
                    top_players = cluster_data.sort_values('similarity', ascending=False).head(3)

                    # Add a text label for each player
                    for _, player in top_players.iterrows():
                        plt.annotate(
                            player['Player'],  # Text to show (player name)
                            (player['pca_1'], player['pca_2']),  # Position of the point
                            xytext=(5, 5),  # Small offset for the text
                            textcoords='offset points',
                            fontsize=annotation_size,
                            alpha=0.7  # Slightly transparent text
                        )

            # Step 6: Add cluster centers to the plot
            # These show the "average" player in each cluster
            plt.scatter(
                centers_pca[:, 0],  # X coordinates of centers
                centers_pca[:, 1],  # Y coordinates of centers
                c='red',  # Red color for centers
                marker='X',  # X shape for centers
                s=200,  # Larger size for centers
                label='Cluster Centers'
            )

            # Step 7: Add role labels to the cluster centers if available
            if role_mapping:
                for i, center in enumerate(centers_pca):
                    if i in role_mapping:
                        plt.annotate(
                            role_mapping[i],  # Role name
                            (center[0], center[1]),  # Position of center
                            xytext=(10, 10),  # Offset for text
                            textcoords='offset points',
                            fontsize=annotation_size + 2,  # Slightly larger font
                            fontweight='bold'  # Bold text for emphasis
                        )

            # Step 8: Add title and axis labels
            explained_var = cluster_results['explained_variance']
            plt.title(f'{position_group} Role Classification')
            # Label axes with how much variation they explain
            plt.xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)')
            plt.ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)')

            # Step 9: Add a legend
            if role_mapping:
                # Create legend elements for each cluster and its role
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=colors[i], markersize=10,
                                            label=f"Cluster {i}: {role}")
                                  for i, role in role_mapping.items()]
                plt.legend(handles=legend_elements)
            else:
                plt.legend()

            # Step 10: Make the layout neat
            plt.tight_layout()

            # Step 11: Convert the plot to a base64 string for web display
            # This allows the plot to be embedded in a web page
            buffer = io.BytesIO()  # Create a buffer to store the image
            plt.savefig(buffer, format='png', dpi=100)  # Save plot to buffer
            buffer.seek(0)  # Go to start of buffer
            img_str = base64.b64encode(buffer.read()).decode('utf-8')  # Encode as base64
            plt.close()  # Close the plot to free memory

            return img_str

        except Exception as e:
            # If something goes wrong, log the error and return None
            logger.error(f"Error generating visualization: {str(e)}")
            return None


    def get_player_percentile_stats(self, player_name: str) -> Dict[str, List[Dict]]:
      
      """
      Generate percentile and raw stat summaries for a player's relevant metrics
      based on position group and most likely role.

      Returns:
          {
              "position_group": [ {name, percentile, rawValue, metric}, ... ],
              "role_group": [ {name, percentile, rawValue, metric}, ... ]
          }
      """
      self._ensure_data_loaded()
      
      # Step 1: Get player row
      player_row = self.data[self.data['Player'] == player_name]
      if player_row.empty:
          raise ValueError(f"Player '{player_name}' not found in dataset.")
      
      player_row = player_row.iloc[0]
      position = player_row['TmPos']
      
      # Step 2: Get roles and most probable role
      role_probs = self.get_role_probabilities(player_name)
      primary_role = max(role_probs.items(), key=lambda x: x[1])[0]
      
      # Step 3: Get relevant metrics
      role_metrics = self.classifier.role_metrics.get(primary_role, [])
      position_group = self.classifier.get_position_group(position)
      
      result = {
          "position_group": [],
        #   "role_group": []
      }

      # Filter players in position group
      pos_group_df = self.data[self.data['TmPos'].isin(position_group)].copy()
    #   role_group_df = pos_group_df.copy()  # Or use full dataset if you want broader comparison
      
      # Step 4: Calculate percentiles
      for metric in role_metrics:
          if metric in self.data.columns:
              player_value = player_row[metric]
              # Calculate percentiles
              pos_vals = pos_group_df[metric].dropna()
            #   role_vals = role_group_df[metric].dropna()

              pos_percentile = round((pos_vals < player_value).mean() * 100, 1)
            #   role_percentile = round((role_vals < player_value).mean() * 100, 1)

              # Get readable name and unit (optional enhancements)
              name = metric.replace("_", " ")
              unit = "per 90" if "per_90" in metric or metric.endswith("_90") else ""

              result["position_group"].append({
                  "name": name,
                  "percentile": pos_percentile,
                  "rawValue": player_value,
                  "metric": unit
              })

            #   result["role_group"].append({
            #       "name": name,
            #       "percentile": role_percentile,
            #       "rawValue": player_value,
            #       "metric": unit
            #   })
      
      return result

    def analyze_player_profile(self, player_name: str) -> Dict[str, Any]:
        """
        Create a comprehensive analysis of a player's profile.

        Parameters:
        - player_name: Name of the player to analyze

        Returns:
        - Dictionary with detailed player analysis results

        What happens:
        - Gets the player's basic information
        - Calculates their role probabilities
        - Finds similar players
        - Finds players in alternative positions with similar style
        - Creates visualizations
        - Returns all analysis in a structured format
        """
        # Step 1: Make sure data is loaded
        self._ensure_data_loaded()

        # Step 2: Find the player in our dataset
        player_data = self.processed_data[self.processed_data['Player'] == player_name]

        # Check if player exists
        if player_data.empty:
            raise ValueError(f"Player {player_name} not found in dataset")

        # Step 3: Get basic information about the player
        position = player_data['TmPos'].iloc[0]  # Player's position

        # Step 4: Calculate role probabilities
        # This tells us what role the player is most suited for
        # For example, is a forward more of a "Finisher" or "Target Man"?
        role_probs = self.get_role_probabilities(player_name, position)

        # Step 5: Find similar players in the same position group
        similar_players = self.find_similar_players(
            target_player=player_name,
            n_recommendations=5,  # Show top 5 similar players
            use_position_groups=True  # Only compare to players in similar positions
        )

        # Step 6: Find alternative position players
        # These are players in different positions with similar playing styles
        alternative_players = self.find_similar_players(
            target_player=player_name,
            n_recommendations=3,  # Show top 3 alternatives
            use_position_groups=False,  # Don't restrict by position group
            alternative_positions=True  # Look for players in other positions
        )

        # Step 7: Build the complete player profile
        # This combines all our analysis into one organized dictionary
        profile = {
            'player_name': player_name,
            'position': position,
            'role_probabilities': role_probs,  # How likely the player fits each role
            'primary_role': max(role_probs.items(), key=lambda x: x[1])[0],  # Most suitable role
            'similar_players': similar_players.to_dict('records'),  # List of similar players
            'alternative_position_players': alternative_players.to_dict('records')  # Players in different positions
        }

        # Step 8: Try to add visualization if matplotlib is available
        # This creates a visual chart showing where the player fits among others
        try:
            # Run clustering for position group
            # This groups players with similar styles into clusters
            cluster_results = self.cluster_players(
                position_group=position,
                n_clusters=len(role_probs)  # Create one cluster per possible role
            )

            # Generate the visualization (creates an image)
            vis_img = self.generate_cluster_visualization(cluster_results)
            if vis_img:
                profile['visualization'] = vis_img  # Add the image to our profile

        except Exception as e:
            # If visualization fails, just log a warning and continue
            # The rest of the analysis is still valuable without the visual
            logger.warning(f"Could not generate visualization: {str(e)}")

        # Step 9: Return the complete profile with all analysis results
        return profile


if __name__ == "__main__":
    # Example usage of the PlayerAnalysis class
    player_analysis = PlayerRecommendationSystem()
    player_analysis.load_data("path_to_your_data.csv")  # Load your data here

    # Analyze a specific player
    profile = player_analysis.analyze_player_profile("Lionel Messi")
    print(profile)  # Print the player's profile analysis