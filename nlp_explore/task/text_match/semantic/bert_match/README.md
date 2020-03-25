使用bert 进行文本匹配任务:

1. 在processor_handler.py 中创建自己的processor, 进行数据处理等操作, 如: `TextMatchProcessor`;

2. 在task.py 中创建自己的task任务;

3. 在run.py 添加任务相关的processor，并修改`create_model` 函数;

4. 修改train.sh 相关的运行参数;
