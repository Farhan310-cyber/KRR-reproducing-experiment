import torch.nn as nn
import torch


class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.args = args

        self.emb_dim = args.dim  # Dimension of the entity/relation embeddings
        self.epsilon = 2.0  # Small epsilon for stability

        self.gamma = torch.Tensor([args.gamma])  # Margin used in scoring functions

        # The range for embedding initialization
        self.embedding_range = torch.Tensor([(self.gamma.item() + self.epsilon) / args.dim])

        # Pi constant used in certain operations
        self.pi = 3.14159265358979323846

    def forward(self, sample, ent_emb, rel_emb, mode='single'):
        '''
        Forward function that calculates the score of a batch of triples.
        Different modes handle positive and negative sample combinations for head-batch, tail-batch, etc.
        '''
        self.entity_embedding = ent_emb  # The entity embeddings
        self.relation_embedding = rel_emb  # The relation embeddings

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            # Select head, relation, and tail from embeddings
            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            if head_part is not None:
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        elif mode == 'rel-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 2]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError(f'mode {mode} not supported')

        # Calculate scores using RotatE model
        score = self.RotatE(head, relation, tail, mode)
        return score

    def RotatE(self, head, relation, tail, mode):
        '''
        RotatE model implementation for score calculation:
        It uses complex numbers for embedding head, tail, and relation.
        '''
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make the relation phases uniformly distributed
        phase_relation = relation / (self.embedding_range.item() / self.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # Scoring calculation
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        # Combine real and imaginary parts
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)  # Calculate the norm of the complex score

        # Apply margin
        score = self.gamma.item() - score.sum(dim=2)
        return score

