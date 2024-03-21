from torch_geometric.data import Data
import copy
import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform


class VirtualNode(BaseTransform):
    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        edge_type = data.get('edge_type', torch.zeros_like(row))
        num_nodes = data.num_nodes
        assert num_nodes is not None

        arange = torch.arange(num_nodes, device=row.device)
        full = row.new_full((num_nodes,), num_nodes)
        row = torch.cat([row, full], dim=0)
        col = torch.cat([col, arange], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        new_type = edge_type.new_full((num_nodes,), int(edge_type.max()) + 1)
        edge_type = torch.cat([edge_type, new_type, new_type + 1], dim=0)

        old_data = copy.copy(data)
        for key, value in old_data.items():
            if key == 'edge_index' or key == 'edge_type':
                continue

            if isinstance(value, Tensor):
                dim = old_data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == 'edge_weight':
                    size[dim] = 2 * num_nodes
                    fill_value = 1.
                elif key == 'batch':
                    size[dim] = 1
                    fill_value = int(value[0])
                elif old_data.is_edge_attr(key):
                    size[dim] = 2 * num_nodes
                    fill_value = 0.
                elif old_data.is_node_attr(key):
                    size[dim] = 1
                    fill_value = 1

                if fill_value is not None:
                    new_value = value.new_full(size, fill_value)
                    data[key] = torch.cat([value, new_value], dim=dim)

        data.edge_index = edge_index
        data.edge_type = edge_type

        if 'num_nodes' in data:
            data.num_nodes = num_nodes + 1

        return data
