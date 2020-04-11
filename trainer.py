import torch
import logging
import models
import numpy as np
import pandas as pd
import os
from utils.misc import average, plot_image_grid
from data import get_loaders
from ast import literal_eval
from torch.optim.lr_scheduler import StepLR
from utils.recorderx import RecoderX
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split

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
        self.scheduler = StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # Initialize loss
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def _init(self):
        # Init parameters
        self.num_train_steps = 0
        self.num_eval_steps = 0
        self.accuracy_best = 0.
        self.losses = {'loss_train': [], 'loss_eval': [], 'accuracy_train': [], 'accuracy_eval': []}

        # Initialize model
        self._init_model()

        # Initialize optimizer
        self._init_optim()

        # Initialize best model
        self.model_best = deepcopy(self.model)

    def _train_iteration(self, data, i, total, correct):
        self.num_train_steps += 1

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Set inputs
        inputs = data['input'].to(self.device)
        targets = data['target'].to(self.device)

        # Forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(outputs.data, dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Record statisctic
        self.losses['loss_train'].append(loss.data.item())
        if self.args.use_tb:
            self.tb.add_scalar('data/loss_train_{}'.format(self.session), self.losses['loss_train'][-1], self.num_train_steps)
        if i % self.args.print_freq == 0:
            line2print = 'Iteration {}, Accuracy: {:.2f}, Loss: {:.4f}'.format(i + 1, 100 * (predicted == targets).sum().item() / targets.size(0), self.losses['loss_train'][-1])
            logging.info(line2print)

        return total, correct

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total, correct = 0., 0.

        # Train over epochs
        for i, data in enumerate(loader):
            total, correct = self._train_iteration(data, i, total, correct)

        # Record statisctic
        self.losses['accuracy_train'].append(100. * correct / total)
        if self.args.use_tb:
            self.tb.add_scalar('data/accuracy_train_{}'.format(self.session), self.losses['accuracy_train'][-1], epoch)

    def _eval_epoch(self, loader, epoch):
        self.model.eval()
        total, correct = 0., 0.

        # Train over epoch
        for i, data in enumerate(loader):
            total, correct = self._eval_iteration(data, total, correct)

        # Record statisctic
        accuracy = 100 * correct / total
        loss = np.array(self.losses['loss_eval'][-len(loader):]).mean()
        self.losses['accuracy_eval'].append(accuracy)

        if self.args.use_tb:
            self.tb.add_scalar('data/accuracy_eval_{}'.format(self.session), self.losses['accuracy_eval'][-1], epoch)
        line2print = 'Evaluation: Accuracy: {:.2f}, Loss: {:.5f}\n'.format(accuracy, loss)
        logging.info(line2print)

        return accuracy

    def _eval_iteration(self, data, total, correct):
        self.num_eval_steps += 1

        # Set inputs
        inputs = data['input'].to(self.device)
        targets = data['target'].to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Record statisctic
        self.losses['loss_eval'].append(loss.data.item())
        if self.args.use_tb:
            self.tb.add_scalar('data/loss_eval_{}'.format(self.session), self.losses['loss_eval'][-1], self.num_eval_steps)

        return total, correct

    def _test_iteration(self, data):
        self.num_eval_steps += 1

        # Set inputs
        inputs = data['input'].to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model_best(inputs)

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
        for i, data in enumerate(loaders['eval']):
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
        name = self.args.model
        torch.save(self.model_best.state_dict(), self.args.save_path + '/' + name + '_s' + str(self.session + 1) + '.pt')

    def _save_submission(self, df, preds):
        # Save data-frame
        df[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds
        df.to_csv(self.args.save_path + '/submission' + '_s{}.csv'.format(self.session), index=False)

    def _train(self, loaders):
        # Initialize
        self._init()

        # Run epoch iterations
        logging.info('\nSession {}'.format(self.session + 1))
        for epoch in range(self.args.epochs):
            logging.info('\nEpoch {}'.format(epoch + 1))

            # Train
            self._train_epoch(loaders['train'], epoch)

            # Eval
            accuracy = self._eval_epoch(loaders['eval'], epoch)

            # Scheduler
            self.scheduler.step(epoch=epoch)

            # Check best model
            if accuracy > self.accuracy_best:
                self.accuracy_best = accuracy
                self.model_best = deepcopy(self.model)

        # Best Score
        logging.info('Best Evaluation Accuracy: {:.2f}\n'.format(self.accuracy_best))

        # Save model
        self._save_model()

        # Update session
        self.session += 1

    def train_cross_validation(self):
        # Set folds
        stratified_k_fold = StratifiedKFold(n_splits=self.args.num_splits, shuffle=True, random_state=self.args.seed)
        df_folds = pd.read_csv(os.path.join(self.args.root, 'train.csv')).iloc[:, 1:].values
        y_folds = df_folds[:, 2] + df_folds[:, 3] * 2 + df_folds[:, 1] * 3

        # Run cross validation
        accuracy = []
        preds_tot = None
        for i, (train_index, eval_index) in enumerate(stratified_k_fold.split(df_folds, y_folds)):
            # Set data-frames
            df = pd.read_csv(os.path.join(self.args.root, 'train.csv'))
            df_train = df.iloc[train_index]
            df_train.reset_index(drop=True, inplace=True)

            df_eval = df.iloc[eval_index]
            df_eval.reset_index(drop=True, inplace=True)

            # Get loaders
            loaders = get_loaders(self.args, df_train, df_eval)

            # Run training
            self._train(loaders)
            accuracy.append(self.accuracy_best)

            # Run test
            preds = self._test_epoch()

            if preds_tot is None:
                preds_tot = preds / self.args.num_splits
            else:
                preds_tot += preds / self.args.num_splits

        logging.info('\nFinished cross-validation with average accuracy of: {:.2f}'.format(average(accuracy)))

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

        logging.info('\nFinished training with accuracy of: {:.2f}'.format(self.accuracy_best))

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