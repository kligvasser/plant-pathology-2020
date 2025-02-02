import torch
import logging
import models
import numpy as np
import pandas as pd
import os
from utils.misc import average
from models.modules.losses import CutMixCrossEntropyLoss, DenseCrossEntropy
from data import get_loaders
from ast import literal_eval
from torch.optim.lr_scheduler import StepLR
from utils.optim import get_exp_scheduler_with_warmup
from utils.recorderx import RecoderX
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.session = 0
        self.print_model = True
        
        if self.args.use_tb:
            self.tb = RecoderX(log_dir=args.save_path)

    def _init_model(self):
        # Initialize model
        if self.args.model_config is not '':
            model_config = dict({}, **literal_eval(self.args.model_config))
        else:
            model_config = {}
        model = models.__dict__[self.args.model]
        self.model = model(**model_config).to(self.device)

        # Print model
        if self.print_model:
            logging.info(self.model)
            logging.info('Number of parameters: {}'.format(sum([l.nelement() for l in self.model.parameters()])))
            self.print_model = False

        # Loading weights
        if self.args.file2load is not '':
            logging.info('\nLoading model...')
            self.model.load_state_dict(torch.load(self.args.file2load))

        # Data parallel
        if len(self.args.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, self.args.device_ids)

    def _init_optim(self):
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        # Initialize scheduler
        if self.args.exp_scheduler:
            self.scheduler = get_exp_scheduler_with_warmup(self.optimizer)
        else:
            self.scheduler = StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # Initialize loss
        if self.args.cutmix:
            self.criterion = CutMixCrossEntropyLoss().to(self.device)
        else:
            self.criterion = DenseCrossEntropy().to(self.device)

    def _init(self):
        # Init parameters
        self.num_train_steps = 0
        self.num_eval_steps = 0
        self.score_best = 0.
        self.losses = {'loss_train': [], 'loss_eval': [], 'accuracy_train': [], 'auc_train': [], 'accuracy_eval': [], 'auc_eval': []}

        # Initialize model
        self._init_model()

        # Initialize optimizer
        self._init_optim()

        # Initialize best model
        self.model_best = deepcopy(self.model)

    def _train_iteration(self, data, i):
        self.num_train_steps += 1

        # Set inputs
        inputs = data['input'].to(self.device)
        targets = data['target'].to(self.device)

        # Forward + backward
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        # Optimize
        if self.num_train_steps % self.args.update_rate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Compute scores
        preds = torch.softmax(outputs, dim=1)
        acc, auc = self._compute_scores(preds.data.cpu(), targets.data.cpu())

        # Record statisctic
        self.losses['loss_train'].append(loss.data.item())
        self.losses['accuracy_train'].append(acc)
        self.losses['auc_train'].append(auc)
        if self.args.use_tb:
            self.tb.add_scalar('data/loss_train_{}'.format(self.session), self.losses['loss_train'][-1], self.num_train_steps)
            self.tb.add_scalar('data/accuracy_train_{}'.format(self.session), self.losses['accuracy_train'][-1], self.num_train_steps)
            self.tb.add_scalar('data/auc_train_{}'.format(self.session), self.losses['auc_train'][-1], self.num_train_steps)
        if i % self.args.print_freq == 0:
            line2print = 'Iteration {}, Accuracy: {:.2f}, AUC: {:.3f}, Loss: {:.4f}'.format(i + 1, acc, auc, self.losses['loss_train'][-1])
            logging.info(line2print)

    def _eval_iteration(self, data, all_preds, all_targets):
        self.num_eval_steps += 1

        # Set inputs
        inputs = data['input'].to(self.device)
        targets = data['target'].to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model(inputs)
            preds = torch.softmax(outputs, dim=1)
            loss = self.criterion(outputs, targets)

        # Save for statisctic
        if all_preds is None:
            all_preds = preds.data.cpu()
            all_targets = targets.data.cpu()
        else:
            all_preds = torch.cat((all_preds, preds.data.cpu()), dim=0)
            all_targets = torch.cat((all_targets, targets.data.cpu()), dim=0)

        # Record statisctic
        self.losses['loss_eval'].append(loss.data.item())
        if self.args.use_tb:
            self.tb.add_scalar('data/loss_eval_{}'.format(self.session), self.losses['loss_eval'][-1], self.num_eval_steps)

        return all_preds, all_targets

    def _train_epoch(self, loader):
        # Zero the parameter gradients
        self.model.train()
        self.optimizer.zero_grad()

        # Train over epochs
        for i, data in enumerate(loader):
            self._train_iteration(data, i)

    def _eval_epoch(self, loader, epoch):
        self.model.eval()
        all_preds, all_targets = None, None

        # Train over epoch
        for i, data in enumerate(loader):
            all_preds, all_targets = self._eval_iteration(data, all_preds, all_targets)

        # Record statisctic
        acc, auc = self._compute_scores(all_preds, all_targets)
        loss = np.array(self.losses['loss_eval'][-len(loader):]).mean()
        self.losses['accuracy_eval'].append(acc)
        self.losses['auc_eval'].append(auc)

        if self.args.use_tb:
            self.tb.add_scalar('data/accuracy_eval_{}'.format(self.session), self.losses['accuracy_eval'][-1], epoch)
            self.tb.add_scalar('data/auc_eval_{}'.format(self.session), self.losses['auc_eval'][-1], epoch)
        line2print = 'Evaluation: Accuracy: {:.2f}, AUC {:.3f}, Loss: {:.5f}\n'.format(acc, auc, loss)
        logging.info(line2print)

        return auc, acc

    def _compute_scores(self, preds, targets):
        # Targets of cutmix
        if targets.ndimension() == 2:
            _, targets = torch.max(targets, 1)

        # Compute accuracy
        _, predicted = torch.max(preds, 1)
        acc = 100. * (predicted == targets).sum() / targets.size(0)

        # Compute AUC
        targets_1hot = torch.zeros_like(preds)
        targets_1hot.scatter_(1, targets.unsqueeze(-1), 1.0)
        try:
            auc = roc_auc_score(targets_1hot, preds, average='macro')
        except ValueError:
            auc = 0.

        return acc, auc

    def _test_iteration(self, data):
        self.num_eval_steps += 1

        # Set inputs
        inputs = data['input'].to(self.device)
        inputs_size = inputs.size()

        # TTA
        if self.args.tta:
            inputs = inputs.view(-1, inputs_size[2], inputs_size[3], inputs_size[4])

        # Run model
        with torch.no_grad():
            outputs = self.model_best(inputs)

        if self.args.tta:
            outputs = outputs.view(inputs_size[0], inputs_size[1], -1).mean(dim=1)

        return outputs.data.cpu()

    def _test_epoch(self):
        # Initialize
        self.model_best.eval()
        preds = None

        # Get loaders
        df = pd.read_csv(os.path.join(self.args.root, 'sample_submission.csv'))
        df.iloc[:, 1:] = 0
        loaders = get_loaders(self.args, df, df)

        # Test over epoch
        for i, data in enumerate(loaders['test']):
            outputs = self._test_iteration(data)

            if preds is None:
                preds = outputs
            else:
                preds = torch.cat((preds, outputs), dim=0)

        # Save submission
        self._save_submission(df, torch.softmax(preds, dim=1))
        return preds

    def _save_model(self):
        # Save models
        torch.save(self.model_best.state_dict(), os.path.join(self.args.save_path, '{}_s{}.pt'.format(self.args.model, self.session + 1)))

    def _save_submission(self, df, preds):
        # Save data-frame
        df[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds
        df.to_csv(os.path.join(self.args.save_path, 'submission_s{}.csv'.format(self.session)), index=False)

    def _train(self, loaders):
        # Initialize
        self._init()

        # Run epoch iterations
        logging.info('\nSession {}'.format(self.session + 1))
        for epoch in range(self.args.epochs):
            logging.info('\nEpoch {}'.format(epoch + 1))

            # Train
            self._train_epoch(loaders['train'])

            # Eval
            auc, acc = self._eval_epoch(loaders['eval'], epoch)
            score = auc * self.args.weight_auc + (acc / 100.) * self.args.weight_acc

            # Check best model
            if score > self.score_best:
                self.score_best = score
                self.model_best = deepcopy(self.model)

            # Scheduler
            self.scheduler.step(epoch=epoch)

        # Best Score
        logging.info('Best evaluation score: {:.3f}\n'.format(self.score_best))

        # Save model
        self._save_model()

        # Update session
        self.session += 1

    def train_cross_validation(self):
        # Set folds
        # train_csv_file = os.path.join(self.args.root, 'train.csv')
        train_csv_file = './scripts/csv/generated/train_100.csv'
        stratified_k_fold = StratifiedKFold(n_splits=self.args.num_splits, shuffle=True, random_state=self.args.seed)
        df_folds = pd.read_csv(train_csv_file).iloc[:, 1:].values
        y_folds = np.argmax(df_folds, axis=1)

        # Run cross validation
        scores = []
        preds_tot = None
        for i, (train_index, eval_index) in enumerate(stratified_k_fold.split(df_folds, y_folds)):
            # Set data-frames
            df = pd.read_csv(train_csv_file)
            df_train = df.iloc[train_index]
            df_train.reset_index(drop=True, inplace=True)

            df_eval = df.iloc[eval_index]
            df_eval.reset_index(drop=True, inplace=True)

            # Get loaders
            loaders = get_loaders(self.args, df_train, df_eval)

            # Run training
            self._train(loaders)
            scores.append(self.score_best)

            # Run test
            preds = self._test_epoch()

            if preds_tot is None:
                preds_tot = preds / self.args.num_splits
            else:
                preds_tot += preds / self.args.num_splits

        logging.info('\nFinished cross-validation with average score of: {:.3f}\n\n'.format(average(scores)))

        # Save submission
        self.session += 1
        df = pd.read_csv(os.path.join(self.args.root, 'sample_submission.csv'))
        self._save_submission(df, torch.softmax(preds_tot, dim=1))

        # Close tensorboard
        if self.args.use_tb:
            self.tb.close()

    def train(self):
        # Get data-frames
        df = pd.read_csv(os.path.join(self.args.root, 'train.csv'))
        df_train, df_eval = train_test_split(df, test_size=0.2, random_state=self.args.seed)
        df_train = df_train.reset_index(drop=True)
        df_eval = df_eval.reset_index(drop=True)

        # Get loaders
        loaders = get_loaders(self.args, df_train, df_eval)

        # Run training
        self._train(loaders)

        logging.info('\nFinished training with auc of: {:.3f}'.format(self.score_best))

        # Test
        self._test_epoch()

        # Close tensorboard
        if self.args.use_tb:
            self.tb.close()

    def train_no_eval(self):
        # Get data-frames
        df = pd.read_csv(os.path.join(self.args.root, 'train.csv'))

        # Get loaders
        loaders = get_loaders(self.args, df, df)

        # Run training
        self._train(loaders)

        logging.info('\nFinished training with auc of: {:.3f}'.format(self.score_best))

        # Test
        self._test_epoch()

        # Close tensorboard
        if self.args.use_tb:
            self.tb.close()

    def test(self):
        # Initialize
        self._init()
        self.model_best = deepcopy(self.model)

        # Test
        self._test_epoch()

        # Close tensorboard
        if self.args.use_tb:
            self.tb.close()