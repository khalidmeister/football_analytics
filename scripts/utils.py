import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter

def shot_map(data, team_name, color='white', edge_color='red', half='First Half'):
    shot_team = data[data.team_name == team_name]
    shot = shot_team[shot_team.shot_outcome_name != 'Goal']
    goal = shot_team[shot_team.shot_outcome_name == 'Goal']

    pitch = VerticalPitch(pitch_type='statsbomb', half=True, goal_type='box', pitch_color='#22312b', line_color='#c7d5cc')
    fig, axs = pitch.grid(figheight=10, title_height=0.08, title_space=0, 
                          endnote_height=0.05, endnote_space=0, grid_height=0.82, axis=False)
    fig.set_facecolor('#22312b')

    shots_scatter = pitch.scatter(shot.x, shot.y, s=(shot.shot_statsbomb_xg * 900) + 100, 
                                  c=color, edgecolors=edge_color, marker='o', ax=axs['pitch'])
    goals_scatter = pitch.scatter(goal.x, goal.y, s=(goal.shot_statsbomb_xg * 900) + 100, 
                                  c=color, edgecolors=edge_color, marker='*', ax=axs['pitch'])
    
    axs['title'].text(0.5, 0.7, 'The Shots Map from ' + team_name, va='center', ha='center', color='#c7d5cc', fontsize=30)
    axs['title'].text(0.5, 0.25, 'The Game\'s ' + half, va='center', ha='center', color='#c7d5cc', fontsize=18)
    

def pressure_heat_map(df, team_name, half='First Half'):
    pressure = df[df.team_name==team_name]

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, 
                  pitch_color='#22312b', line_color='#efefef')
    fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0, axis=False,
                        title_space=0, grid_height=0.82, endnote_height=0.05)
    fig.set_facecolor('#22312b')

    bin_statistic = pitch.bin_statistic(pressure.x, pressure.y, 
                                        statistic='count', bins=(25, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)

    pcm = pitch.heatmap(bin_statistic, ax=axs['pitch'], cmap='hot', edgecolors='#22312b')

    cbar = fig.colorbar(pcm, ax=axs['pitch'], shrink=0.6)
    cbar.outline.set_edgecolor('#efefef')
    cbar.ax.yaxis.set_tick_params(color='#efefef')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

    axs['title'].text(0.5, 0.7, 'The Pressure\'s Heat Map from ' + team_name, color='#c7d5cc',
                       va='center', ha='center', fontsize=30)
    axs['title'].text(0.5, 0.0, 'The Game\'s ' + half, color='#c7d5cc',
                       va='center', ha='center', fontsize=18)

    axs['endnote'].text(0.8, 0.5, 'By Irfan Khalid (@MixtureModels)', color='#c7d5cc',
                       va='center', ha='center', fontsize=15)

    plt.show()
    

def voronoi_goal_diagram(goal_data, team_name, player_name, home_color, away_color):
    x = []
    y = []
    team_mate = []
    scorer = []

    for i, data in goal_data.iterrows():
        x.append(data.location[0])
        y.append(data.location[1])
        team_mate.append(True)
        scorer.append(True)   
        for j in data.shot_freeze_frame:
            x.append(j['location'][0])
            y.append(j['location'][1])
            team_mate.append(j['teammate'])
            scorer.append(False)

    x = np.array(x)
    y = np.array(y)
    team_mate = np.array(team_mate)
    scorer = np.array(scorer)

    pitch = VerticalPitch(pitch_type='statsbomb', half=True, pitch_color='#22312b', line_color='#efefef')
    fig, axs = pitch.grid(figheight=8, title_height=0.08, title_space=0, 
                         endnote_height=0.05, endnote_space=0, grid_height=0.82, axis=False)
    fig.set_facecolor('#22312b')

    team_1, team_2 = pitch.voronoi(x, y, team_mate)

    pol_1 = pitch.polygon(team_1, ax=axs['pitch'], fc=home_color, ec='white', lw=3, alpha=0.4)
    pol_2 = pitch.polygon(team_2, ax=axs['pitch'], fc=away_color, ec='white', lw=3, alpha=0.4)

    sca_1 = pitch.scatter(x[team_mate], y[team_mate], ax=axs['pitch'], c=home_color, s=150)
    sca_2 = pitch.scatter(x[~team_mate], y[~team_mate], ax=axs['pitch'], c=away_color, s=150)

    sca_3 = pitch.scatter(x[scorer], y[scorer], ax=axs['pitch'], c='white', s=150)

    axs['title'].text(0.5, 0.70, 'The Voronoi Diagram of ' + team_name +'\'s ' + 'Goal', 
                      color='#efefef', va='center', ha='center', fontsize=27)
    axs['title'].text(0.5, 0, 'By ' + player_name,
                     color='#efefef', va='center', ha='center', fontsize=18)

    axs['endnote'].text(0.85, 0.5, 'By Irfan Khalid (@MixtureModels)',
                       color='#efefef', va='center', ha='center', fontsize=10)


def shot_progress_chart(data, home_team, away_team, event_index=None):
    if event_index == None:
        shots = data
    else:
        shots = data.loc[:event_index]
    
    shots = shots[data.type_name == 'Shot']
    shots = shots[['team_name', 'player_name', 'minute', 'second', 'location', 'shot_statsbomb_xg', 'shot_outcome_name']]
        
    h_xg = []
    a_xg = []
    h_min = []
    a_min = []

    for i, data in shots.iterrows():
        if data.team_name == home_team:
            h_xg.append(data.shot_statsbomb_xg)
            h_min.append(data.minute)
        else:
            a_xg.append(data.shot_statsbomb_xg)
            a_min.append(data.minute)

    h_xg = np.array(h_xg)
    a_xg = np.array(a_xg)

    cum_h_xg = np.round(np.cumsum(h_xg), 2)
    cum_a_xg = np.round(np.cumsum(a_xg), 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor('#22312b')
    ax.patch.set_facecolor('#22312b')

    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_alpha(0)
    ax.spines['right'].set_alpha(0)

    ax.tick_params(colors='white')

    ax.step(x=h_min, y=cum_h_xg, label=home_team)
    ax.step(x=a_min, y=cum_a_xg, label=away_team)

    ax.set_xlabel('Minute')
    ax.set_ylabel('xG')

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.set_title('The xG Progress Chart between ' + home_team + ' and ' + away_team)
    ax.title.set_color('white')

    plt.legend()
    plt.show()