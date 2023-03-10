# coding=utf-8
from __future__ import print_function
import sys
import os

import numpy as np
import paddle
import paddle.nn.functional as F

import traceback
import importlib, argparse
from datetime import datetime
import multiprocessing
from algs.performance_pred.utils import TrainConfig
from ast import literal_eval
from comm.registry import Registry
from train.utils import OptimizerConfig, LRConfig

from compute.redis import RedisLog
from compute.pid_manager import PIDManager


class TrainModel(object):
    def __init__(self, file_id, logger):

        # module_name = 'scripts.%s'%(file_name)
        module_name = file_id
        if module_name in sys.modules.keys():
            self.log_record('Module:%s has been loaded, delete it' % (module_name))
            del sys.modules[module_name]
            _module = importlib.import_module('.', module_name)
        else:
            _module = importlib.import_module('.', module_name)

        net = _module.EvoCNNModel()
        best_acc = 0.0
        self.net = net

        TrainConfig.ConfigTrainModel(self)
        # initialize optimizer
        o = OptimizerConfig()
        opt_cls = Registry.OptimizerRegistry.query(o.read_ini_file('_name'))
        opt_params = {k: v for k, v in o.read_ini_file_all().items() if not k.startswith('_')}
        l = LRConfig()
        lr_cls = Registry.LRRegistry.query(l.read_ini_file('lr_strategy'))
        lr_params = {k: v for k, v in l.read_ini_file_all().items() if not k.startswith('_')}
        lr_params['lr'] = float(lr_params['lr'])
        opt_params['lr'] = float(lr_params['lr'])
        self.opt_params = opt_params
        self.opt_cls = opt_cls
        self.opt_params['total_epoch'] = self.nepochs
        self.lr_params = lr_params
        self.lr_cls = lr_cls
        # after the initialization

        self.best_acc = best_acc

        self.file_id = file_id
        self.logger = logger

    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('[%s]-%s' % (dt, _str))

    def get_optimizer(self, epoch):
        # get optimizer
        self.opt_params['current_epoch'] = epoch
        opt_cls_ins = self.opt_cls(**self.opt_params)
        optimizer = opt_cls_ins.get_optimizer(self.net.parameters())
        return optimizer

    def get_learning_rate(self, epoch):
        self.lr_params['optimizer'] = self.get_optimizer(epoch)
        self.lr_params['current_epoch'] = epoch
        lr_cls_ins = self.lr_cls(**self.lr_params)
        learning_rate = lr_cls_ins.get_learning_rate()
        return learning_rate

    def train(self, epoch):
        self.net.train()
        optimizer = self.get_optimizer(epoch)
        train_loss = list()
        train_acc = list()
        for _, data in enumerate(self.trainloader()):
            inputs, labels = data
            inputs = paddle.to_tensor(inputs)
            labels = paddle.to_tensor(labels)
            labels = paddle.reshape(labels, [-1, 1])

            predicts = self.net(inputs)
            loss = F.cross_entropy(predicts, labels)
            train_loss.append(loss.numpy())
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            acc = paddle.metric.accuracy(input=predicts, label=labels)
            train_acc.append(acc.numpy())
        loss_mean = np.mean(train_loss)
        acc_mean = np.mean(train_acc)
        self.log_record('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f' % (epoch + 1, loss_mean, acc_mean))

    def main(self, epoch):
        self.net.eval()
        test_loss = []
        test_acc = []
        for _, data in enumerate(self.validate_loader()):
            inputs, labels = data
            inputs = paddle.to_tensor(inputs)
            labels = paddle.to_tensor(labels)
            labels = paddle.reshape(labels, [-1, 1])
            outputs = self.net(inputs)
            loss = F.cross_entropy(outputs, labels)
            test_loss.append(loss.numpy())
            acc = paddle.metric.accuracy(input=outputs, label=labels)
            test_acc.append(acc.numpy())
        avg_loss = np.mean(test_loss)
        avg_acc = np.mean(test_acc)
        if avg_acc > self.best_acc:
            self.best_acc = avg_acc
        self.log_record('Valid-Epoch:%3d, Loss:%.3f, Acc:%.3f' % (epoch + 1, avg_loss, avg_acc))

    def process(self):
            total_epoch = self.nepochs
            scheduler = self.get_learning_rate(total_epoch)
            for p in range(total_epoch):
                scheduler.step()
                self.train(p)
                self.main(p)
            return self.best_acc


class RunModel(object):

    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('[%s]-%s' % (dt, _str))

    def do_work(self, gpu_id, file_id, uuid):
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        paddle.set_device('gpu:%s' % gpu_id)
        logger = RedisLog(os.path.basename(file_id) + '.txt')
        best_acc = 0.0
        try:
            m = TrainModel(file_id, logger)
            m.log_record(
                'Used GPU#%s, worker name:%s[%d]' % (gpu_id, multiprocessing.current_process().name, os.getpid()))
            best_acc = m.process()
        except BaseException as e:
            msg = traceback.format_exc()
            print('Exception occurs, file:%s, pid:%d...%s' % (file_id, os.getpid(), str(e)))
            print('%s' % (msg))
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = 'Exception occurs:%s' % (msg)
            logger.info('[%s]-%s' % (dt, _str))
        finally:
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = 'Finished-Acc:%.3f' % best_acc
            logger.info('[%s]-%s' % (dt, _str))

            logger.write_file('RESULTS', 'results.txt', '%s=%.5f\n' % (file_id, best_acc))
            _str = '%s;%.5f\n' % (uuid, best_acc)
            logger.write_file('CACHE', 'cache.txt', _str)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("-gpu_id", "--gpu_id", help="GPU ID", type=str)
    _parser.add_argument("-file_id", "--file_id", help="file id", type=str)
    _parser.add_argument("-uuid", "--uuid", help="uuid of the individual", type=str)

    _parser.add_argument("-super_node_ip", "--super_node_ip", help="ip of the super node", type=str)
    _parser.add_argument("-super_node_pid", "--super_node_pid", help="pid on the super node", type=int)
    _parser.add_argument("-worker_node_ip", "--worker_node_ip", help="ip of this worker node", type=str)

    _args = _parser.parse_args()

    PIDManager.WorkerEnd.add_worker_pid(_args.super_node_ip, _args.super_node_pid, _args.worker_node_ip)

    RunModel().do_work(_args.gpu_id, _args.file_id, _args.uuid)

    PIDManager.WorkerEnd.remove_worker_pid(_args.super_node_ip, _args.super_node_pid, _args.worker_node_ip)
