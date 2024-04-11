import networkx as nx


def test_loop(env, agent, NUM_TEST_EPS):
    reward_history = []
    ep_score_history = []
    actions = []
    for ep in range(NUM_TEST_EPS):
        ep_score = 0
        done = False
        state, info = env.reset(ep)
        while not done:
            mask = info['node_action_mask']
            action = agent.select_action(state, mask)
            actions.append(action)
            next_state, reward, done, truncated, _ = env.step(action)
            ep_score += reward
            state = next_state
            reward_history.append(reward)
        ep_score_history.append(ep_score)

    return actions, reward_history, ep_score_history


def hda(anc, NUM_TEST_EPS, data_path, file_name= None):
    HDA_reward_history = []
    HDA_actions = []
    HDA_ep_score_history = []

    for i in range(NUM_TEST_EPS):
        if file_name is not None:
            graph = nx.read_gml(str(data_path) + '/' + str(file_name))
        else:
            graph = nx.read_gml(str(data_path) + "/g_{}".format(i))

        wt = nx.get_node_attributes(graph, 'weight')
        total_wt = sum(wt.values())
        ep_score = 0
        max_degree_centrality_nodes = []

        length = int(len(graph.nodes))

        while len(graph.edges) > 0:
            degree_centrality = nx.degree_centrality(graph)

            max_degree_centrality_node = max(degree_centrality, key=degree_centrality.get)
            max_degree_centrality_nodes.append(max_degree_centrality_node)
            graph.remove_node(str(max_degree_centrality_node))

            reward = connectivity(anc, graph, wt, max_degree_centrality_node, length, total_wt)
            ep_score += reward
            HDA_reward_history.append(reward)
            HDA_actions.append(max_degree_centrality_node)

        HDA_ep_score_history.append(ep_score)

    return HDA_actions, HDA_reward_history, HDA_ep_score_history


def connectivity(anc, graph, wt, max_degree_centrality_node, length, total_wt):

    GCC = sorted(nx.connected_components(graph), key=len, reverse=True)

    if anc == 'cn':
        cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC]
        denominator = ((length * (length - 1)) / 2) * length
        return sum(cn) / denominator

    elif anc == 'dw_cn':
        weight = wt[str(max_degree_centrality_node)]
        cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC] / total_wt
        denominator = (length * (length - 1)) / 2
        return (sum(cn) * weight) / denominator

    elif anc == 'rw_cn':
        weight = wt[str(max_degree_centrality_node)] / total_wt
        cn = [(len(gcc) * (len(gcc) - 1)) / 2 for gcc in GCC]
        denominator = (length * (length - 1)) / 2
        return (sum(cn) * weight) / denominator

    elif anc == 'nd':
        denominator = length * length
        return len(GCC[0]) / denominator

    elif anc == 'dw_nd':
        weight = wt[str(max_degree_centrality_node)] / total_wt
        denominator = length
        return (len(GCC[0]) * weight) / denominator

    else:
        weight = wt[str(max_degree_centrality_node)] / total_wt
        denominator = length
        return (len(GCC[0]) * weight) / denominator
