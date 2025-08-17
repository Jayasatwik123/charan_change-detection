from tensorboardX import SummaryWriter

class Visualization:
    def __init__(self):
        self.writer = None # Initialize to None instead of empty string

    def create_summary(self, model_type='U_Net'):
        """新建writer 设置路径"""
        # self.writer = SummaryWriter(model_type, comment=model_type)
        self.writer = SummaryWriter(comment='-' +model_type)

    def add_scalar(self, epoch, value, params='loss'):
        """添加训练记录"""
        if self.writer: # Add a check to ensure writer is initialized
            self.writer.add_scalar(params, value, global_step=epoch)
        else:
            print("Warning: SummaryWriter not initialized. Call create_summary() first.")

    def add_image(self, tag, img_tensor):
        """添加tensor影像"""
        if self.writer: # Add a check
            self.writer.add_image(tag, img_tensor)
        else:
            print("Warning: SummaryWriter not initialized. Call create_summary() first.")

    def add_graph(self, model, input_to_model=None):
        """添加模型图"""
        if self.writer: # Add a check
            if input_to_model is not None:
                self.writer.add_graph(model, input_to_model)
            else:
                # Try to add graph without input, or log a warning
                try:
                    self.writer.add_graph(model)
                except Exception as e:
                    print(f"Warning: Could not add graph. Error: {e}. Consider passing sample input to add_graph.")
        else:
            print("Warning: SummaryWriter not initialized. Call create_summary() first.")

    def close_summary(self):
        """关闭writer"""
        self.writer.close()