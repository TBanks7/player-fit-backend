from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import logging
import base64
from typing import List, Dict, Any, Optional





# Import the player analysis system classes
from player_analysis import PlayerRecommendationSystem, PlayerRoleClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
# Enable CORS to allow requests from your Next.js frontend
CORS(app)

# Initialize the player recommendation system
# We'll load the data when the application starts
DATA_PATH = "data.csv"
player_system = None

try:
    player_system = PlayerRecommendationSystem(DATA_PATH)
    logger.info(f"Successfully loaded player data from {DATA_PATH}")
except Exception as e:
    logger.error(f"Failed to load player data: {str(e)}")

# Helper function to convert DataFrame to JSON-serializable format
def dataframe_to_json(df):
    if df is None:
        return []
    return df.to_dict('records')

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    if player_system is not None and player_system.data is not None:
        return jsonify({
            "status": "healthy",
            "players_loaded": len(player_system.data)
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "message": "Player data not loaded"
        }), 500

@app.route('/players', methods=['GET'])
def get_players():
    """Get a list of all players in the system"""
    if player_system is None:
        return jsonify({"error": "Player system not initialized"}), 500
    
    # Get query parameters for filtering
    position = request.args.get('position')
    team = request.args.get('team')
    name_search = request.args.get('search')
    
    # Filter the data based on parameters
    filtered_data = player_system.data.copy()
    
    if position:
        filtered_data = filtered_data[filtered_data['TmPos'] == position]
    
    if team and 'Squad' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Squad'] == team]
    
    if name_search:
        filtered_data = filtered_data[filtered_data['Player'].str.contains(name_search, case=False)]
    
    # Select only essential columns to reduce response size
    columns = ['Player', 'TmPos']
    if 'Squad' in filtered_data.columns:
        columns.append('Squad')
    if 'Age' in filtered_data.columns:
        columns.append('Age')
    
    # Limit to first 100 players to avoid huge responses
    result = filtered_data[columns].head(2704).to_dict('records')
    
    return jsonify({
        "count": len(result),
        "players": result
    })

@app.route('/positions', methods=['GET'])
def get_positions():
    """Get all available positions in the system"""
    if player_system is None:
        return jsonify({"error": "Player system not initialized"}), 500
    
    positions = player_system.data['TmPos'].unique().tolist()
    
    # Get roles for each position
    role_classifier = PlayerRoleClassifier()
    position_roles = {}
    for position in positions:
        position_roles[position] = role_classifier.get_roles_for_position(position)
    
    return jsonify({
        "positions": positions,
        "position_roles": position_roles
    })

@app.route('/player/<player_name>', methods=['GET'])
def get_player_profile(player_name):
    """Get detailed profile for a specific player"""
    if player_system is None:
        return jsonify({"error": "Player system not initialized"}), 500
    
    try:
        # Check if player exists
        if player_name not in player_system.data['Player'].values:
            return jsonify({"error": f"Player '{player_name}' not found"}), 404
        
        # Get comprehensive player profile
        profile = player_system.analyze_player_profile(player_name)
        
        # Convert any DataFrame objects to lists
        if 'similar_players' in profile and not isinstance(profile['similar_players'], list):
            profile['similar_players'] = dataframe_to_json(profile['similar_players'])
        
        if 'alternative_position_players' in profile and not isinstance(profile['alternative_position_players'], list):
            profile['alternative_position_players'] = dataframe_to_json(profile['alternative_position_players'])
        
        return jsonify(profile)
    
    except Exception as e:
        logger.error(f"Error getting player profile: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/similar-players', methods=['GET'])
def find_similar_players():
    """Find players similar to a specified player"""
    if player_system is None:
        return jsonify({"error": "Player system not initialized"}), 500
    
    player_name = request.args.get('player')
    if not player_name:
        return jsonify({"error": "Missing required parameter: player"}), 400
    
    # Get optional parameters
    count = int(request.args.get('count', 20))
    exclude_team = request.args.get('exclude_team', 'false').lower() == 'true'
    use_positions = request.args.get('use_positions', 'true').lower() == 'true'
    alternative_positions = request.args.get('alternative_positions', 'false').lower() == 'true'
    
    try:
        # Find similar players
        similar_players = player_system.find_similar_players(
            target_player=player_name,
            n_recommendations=count,
            use_position_groups=use_positions,
            exclude_current_team=exclude_team,
            alternative_positions=alternative_positions
        )
        
        # Convert result to JSON
        result = dataframe_to_json(similar_players)
        
        return jsonify({
            "player": player_name,
            "similar_players": result
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error finding similar players: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/cluster-players', methods=['GET'])
def cluster_players():
    """Group players into clusters based on their playing styles"""
    if player_system is None:
        return jsonify({"error": "Player system not initialized"}), 500
    
    position = request.args.get('position')
    if not position:
        return jsonify({"error": "Missing required parameter: position"}), 400
    
    # Get optional parameters
    clusters = int(request.args.get('clusters', 0))
    filter_positions = request.args.get('filter_positions', 'true').lower() == 'true'
    
    try:
        role_classifier = PlayerRoleClassifier()
        possible_roles = role_classifier.get_roles_for_position(position)
        
        # If clusters not specified, use number of possible roles
        if clusters <= 0 and possible_roles:
            clusters = len(possible_roles)
        elif clusters <= 0:
            clusters = 3  # Default to 3 clusters
        
        # Perform clustering
        cluster_results = player_system.cluster_players(
            position_group=position,
            n_clusters=clusters,
            filter_positions=filter_positions
        )
        
        # Generate visualization
        visualization = player_system.generate_cluster_visualization(cluster_results)
        
        # Prepare response data
        result = {
            "position": position,
            "clusters": clusters,
            "visualization": visualization,
            "role_mapping": cluster_results['role_mapping']
        }
        
        # Add cluster statistics
        cluster_stats = {}
        for cluster_id, stats in cluster_results['cluster_stats'].items():
            cluster_stats[str(cluster_id)] = {
                "size": stats['size'],
                "top_players": stats['top_players']
            }
        
        result["cluster_stats"] = cluster_stats
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error clustering players: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/percentile-rank', methods=['GET']) 
def percentile_rank():
    """
    Example request: /percentile-rank?player=Lionel%20Messi
    """
    player_name = request.args.get('player')

    if not player_name:
        return jsonify({"error": "Missing 'player' query parameter"}), 400

    try:
        result = player_system.get_player_percentile_stats(player_name)
        return jsonify(convert_numpy_types(result)), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

@app.route('/role-analysis', methods=['GET'])
def analyze_player_role():
    """Analyze which role a player is best suited for"""
    if player_system is None:
        return jsonify({"error": "Player system not initialized"}), 500
    
    player_name = request.args.get('player')
    if not player_name:
        return jsonify({"error": "Missing required parameter: player"}), 400
    
    # Get optional parameter
    position = request.args.get('position')  # Optional override position
    
    try:
        # Get role probabilities
        role_probs = player_system.get_role_probabilities(player_name, position)
        
        # Sort roles by probability
        sorted_roles = [{"role": role, "probability": prob} 
                      for role, prob in sorted(role_probs.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)]
        
        # Get the player's current position if not specified
        if not position and player_name in player_system.data['Player'].values:
            position = player_system.data[player_system.data['Player'] == player_name]['TmPos'].iloc[0]

        # Get other player profile information
        height = player_system.data[player_system.data['Player'] == player_name]['player_height_mtrs'].iloc[0]
        age = player_system.data[player_system.data['Player'] == player_name]['Age'].iloc[0]
        shirt_number = player_system.data[player_system.data['Player'] == player_name]['player_num'].iloc[0]
        team = player_system.data[player_system.data['Player'] == player_name]['Squad'].iloc[0]
        comp = player_system.data[player_system.data['Player'] == player_name]['Comp'].iloc[0]
        foot = player_system.data[player_system.data['Player'] == player_name]['player_foot'].iloc[0]
        nation = player_system.data[player_system.data['Player'] == player_name]['Nation'].iloc[0]
        market_value = player_system.data[player_system.data['Player'] == player_name]['player_market_value_euro'].iloc[0]
        dob = player_system.data[player_system.data['Player'] == player_name]['DOB'].iloc[0]
        url = player_system.data[player_system.data['Player'] == player_name]['Url'].iloc[0]
        
        return jsonify({
            "player": player_name,
            "team": team,
            "comp": comp,
            "foot": foot,
            "dob": dob,
            "height": height,
            "age": str(age),
            "shirt_number": str(shirt_number),
            "nation": nation,
            "market_value": market_value,
            "url": url,
            "position": position,
            "roles": sorted_roles,
            "primary_role": sorted_roles[0]['role'] if sorted_roles else None
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error analyzing player role: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/squad-builder', methods=['GET'])
def build_squad():
    """Find players for specific roles to build a squad"""
    if player_system is None:
        return jsonify({"error": "Player system not initialized"}), 500
    
    position = request.args.get('position')
    role = request.args.get('role')
    
    if not position or not role:
        return jsonify({"error": "Missing required parameters: position and role"}), 400
    
    # Get optional parameters
    count = int(request.args.get('count', 5))
    
    try:
        # Get all players in this position
        position_players = player_system.data[player_system.data['TmPos'] == position]
        
        if len(position_players) == 0:
            return jsonify({"error": f"No players found for position: {position}"}), 404
        
        # Calculate role probability for each player
        role_scores = {}
        for player_name in position_players['Player'].values:
            try:
                # Calculate role probabilities for this player
                role_probs = player_system.get_role_probabilities(player_name)
                # Store the probability for the specified role
                role_scores[player_name] = role_probs.get(role, 0)
            except:
                # Skip any players that cause errors
                continue
        
        # Sort players by their role probability
        sorted_players = [{"player": player, "score": score} 
                         for player, score in sorted(role_scores.items(), 
                                                    key=lambda x: x[1], 
                                                    reverse=True)]
        
        return jsonify({
            "position": position,
            "role": role,
            "players": sorted_players[:count]
        })
        
    except Exception as e:
        logger.error(f"Error building squad: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)