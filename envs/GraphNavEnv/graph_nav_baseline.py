import networkx as nx


class ShortestPathAgent:

    def __init__(self, save_path=False):
        self.save_path = save_path
        self.path = None

    def get_action(self, obs, info):
        agent_node = info["agent_node"]
        goal_node = info["goal_node"]
        if self.save_path:
            if self.path is None:
                self.path = nx.shortest_path(obs, source=agent_node, target=goal_node, weight="length")
                # remove start node from path
                self.path = self.path[1:]
            return self.path.pop()
        else:
            path = nx.shortest_path(obs, source=agent_node, target=goal_node, weight="length")
            return path[1]
