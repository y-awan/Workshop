import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# ==========================================
# TASK 1: THE BUG HUNT
# Goal: Find and delete the 3 structural flaws in this model.
# ==========================================

class BuggyTwoTowerDLRM(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32):
        super().__init__()
        # User Tower
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # Item Tower
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=4)
        
    def forward(self, user_ids, item_ids):
        # User projection
        u = self.user_mlp(self.user_embed(user_ids))
        u = F.normalize(u, p=2, dim=-1)
        
        u = torch.matmul(torch.eye(u.size(0)), u)
        
        # Item projection
        v = self.item_mlp(self.item_embed(item_ids))
        v = F.normalize(v, p=2, dim=-1)
        
        v = v.squeeze()
        
        # Calculate batch affinity matrix
        similarity_matrix = torch.matmul(u, v.T)
        return similarity_matrix


# ==========================================
# ADVANCED: SITUATIONAL CONDITIONALS
# Reference code for injecting dynamic context into the User Tower
# ==========================================

class TwoTowerWithContext(nn.Module):
    def __init__(self, num_users, num_items, num_contexts, embed_dim=32):
        super().__init__()
        
        # USER TOWER (Condition-Aware)
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.context_embed = nn.Embedding(num_contexts, embed_dim)
        
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), # Accepts concatenated width
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # ITEM TOWER (Static)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, user_ids, context_ids, item_ids):
        # 1. Condition the user vector via concatenation
        u_combined = torch.cat([
            self.user_embed(user_ids), 
            self.context_embed(context_ids)
        ], dim=-1)
        
        # 2. Project and normalize
        u_final = F.normalize(self.user_mlp(u_combined), p=2, dim=-1)
        v_final = F.normalize(self.item_mlp(self.item_embed(item_ids)), p=2, dim=-1)
        
        # 3. Final Affinity Matrix
        return torch.matmul(u_final, v_final.T)


# ==========================================
# INTERACTIVE DEMO: SEMANTIC VECTOR SEARCH
# Run this cell after training to test the model's geometry
# ==========================================

def interactive_movie_recommendation(model, movies_df, top_k=5):
    # 1. Accept dynamic user input
    query = input('\n🍿 Enter a movie you love: ')
    
    # 2. Fuzzy text match to find the closest Movie ID
    matches = movies_df[movies_df['title'].str.contains(query, case=False)]
    
    if len(matches) == 0:
        print(f"❌ Couldn't find '{query}'. Try a classic 90s/80s movie!")
        return
        
    idx = matches.index[0]
    matched_title = movies_df.iloc[idx]['title']
    print(f'\n✅ Matched your query with: {matched_title}')
    
    # 3. Extract normalized vectors directly from trained Item Tower
    with torch.no_grad():
        all_ids = torch.tensor(movies_df.index.values)
        item_vectors = F.normalize(model.item_mlp(model.item_embed(all_ids)), p=2, dim=-1)
        target_vector = item_vectors[idx].unsqueeze(0)
        
        # 4. Dot product against the entire movie catalog
        sims = torch.matmul(target_vector, item_vectors.T).squeeze(0)
        top_matches = torch.topk(sims, k=top_k + 1).indices.numpy()[1:]
        
    print(f"\n🎯 Because you liked {matched_title}, the Two-Tower model recommends:")
    for rank, match_id in enumerate(top_matches, 1):
        print(f'  {rank}. {movies_df.iloc[match_id]["title"]} (Affinity: {sims[match_id]:.4f})')
