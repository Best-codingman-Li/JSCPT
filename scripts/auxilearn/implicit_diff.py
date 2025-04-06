'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-07-11 02:09:17
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-07-24 07:15:10
FilePath: /Prompt/mvlpt-master/scripts/auxilearn/implicit_diff.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
import datetime

class Hypergrad:
    """Implicit differentiation for auxiliary parameters.
    This implementation follows the Algs. in "Optimizing Millions of Hyperparameters by Implicit Differentiation"
    (https://arxiv.org/pdf/1911.02590.pdf), with small differences.

    """

    def __init__(self, learning_rate=.1, truncate_iter=3):
        self.learning_rate = learning_rate
        self.truncate_iter = truncate_iter

    def grad(self, loss_val, loss_train, aux_params, params):
        """Calculates the gradients w.r.t \phi dloss_aux/dphi, see paper for details

        :param loss_val:
        :param loss_train:
        :param aux_params:
        :param params:
        :return:
        """
        # dentropy_dparams = torch.autograd.grad(
        #     entropy_loss,
        #     aux_params,
        #     retain_graph=True,
        #     allow_unused=True
        # )

        dloss_val_dparams = torch.autograd.grad(
            loss_val,
            params,
            retain_graph=True,
            allow_unused=True
        )

        dloss_train_dparams = torch.autograd.grad(
                loss_train,
                params,
                allow_unused=True,
                create_graph=True,
        )

        # exit()

        v2 = self._approx_inverse_hvp(dloss_val_dparams, dloss_train_dparams, params)

        #print("v2.shape", np.array(v2).shape)
        #print("after v2 dloss_val_dparams.shape", np.array(dloss_val_dparams).shape)
       
        for dloss_train_dparam in dloss_train_dparams:
            #print("after v2 dloss_train_dparams.requires_grad", dloss_train_dparam.requires_grad)
            if not dloss_train_dparam.requires_grad:
                dloss_train_dparam.requires_grad = True
        #print("after v2 aux_params", aux_params)

        v3 = torch.autograd.grad(
            dloss_train_dparams,
            aux_params,
            grad_outputs=v2,
            allow_unused=True
        )
        
        #print("this is v3:", v3)
        # note we omit dL_v/d_lambda since it is zero in our settings
        #return list(-g+v for (g,v) in zip(v3,dentropy_dparams))
        #return list(-g for g in v3)
        return list(-g for g in v3 if g)

    # 共轭梯度法的迭代方法来计算逆Hessian向量积的近似结果
    def _approx_inverse_hvp(self, dloss_val_dparams, dloss_train_dparams, params):
        """

        :param dloss_val_dparams: dL_val/dW
        :param dloss_train_dparams: dL_train/dW
        :param params: weights W
        :return: dl_val/dW * dW/dphi
        """
        p = v = dloss_val_dparams
        for i in range(self.truncate_iter):

            # print("Current time:", datetime.datetime.now())
            grad = torch.autograd.grad(
                    dloss_train_dparams,
                    params,
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=True
                )
            # print("Current time:", datetime.datetime.now())

            grad = [g * v_element for g, v_element in zip(grad, v)]

            grad = [g * self.learning_rate for g in grad]  # scale: this a is key for convergence

            v = [curr_v - curr_g for (curr_v, curr_g) in zip(v, grad)]
            # note: different than the pseudo code in the paper
            p = [curr_p + curr_v for (curr_p, curr_v) in zip(p, v)]
        return list(pp for pp in p)
