import sys
import logging
from zai import ZhipuAiClient

# 配置日志
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zhipu_api.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 初始化客户端
client = ZhipuAiClient(api_key="你的API密钥")
logger.info("ZhipuAI客户端初始化完成")

def chat_with_zhipuai(user_input):
    logger.info(f"开始处理用户输入: {user_input[:50]}...")  # 只记录前50个字符
    
    try:
        logger.debug("准备发送API请求")
        logger.debug(f"请求参数: model=glm-4.5-flash, max_tokens=4096, temperature=0.6")
        
        response = client.chat.completions.create(
            model="glm-4.5-flash",
            messages=[{"role": "user", "content": user_input}],
            stream=True,  # 流式响应
            max_tokens=4096,
            temperature=0.6
        )
        logger.info("API请求发送成功，开始接收流式响应")
        
        full_response = ""
        chunk_count = 0
        
        for chunk in response:
            chunk_count += 1
            logger.debug(f"接收到第 {chunk_count} 个数据块")
            
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                logger.debug(f"数据块内容: {delta}")
                logger.debug(f"delta类型: {type(delta)}")
                # 提取content和reasoning_content的值

                content_value = delta.get("content")
                reasoning_value = delta.get("reasoning_content")                
                logger.debug(f"content值: {repr(content_value)}, reasoning_content值: {repr(reasoning_value)}")                

                # 检查content字段是否有有效内容
                if content_value is not None and content_value != "":
                    full_response += content_value
                    print(content_value, end="", flush=True)
                    logger.debug(f"输出content内容: {content_value}")
                    continue
                
                # 检查reasoning_content字段是否有有效内容
                if reasoning_value is not None and reasoning_value != "":
                    full_response += reasoning_value
                    print(reasoning_value, end="", flush=True)
                    logger.debug(f"输出reasoning_content内容: {reasoning_value}")
                    continue
                    
                logger.debug("数据块中无有效内容")
            else:
                logger.warning(f"第 {chunk_count} 个数据块格式异常: {chunk}")
        
        logger.info(f"响应接收完成，总数据块数: {chunk_count}")
        logger.info(f"完整响应长度: {len(full_response)} 字符")
        return full_response
        
    except Exception as e:
        error_message = f"请求出错: {str(e)}"
        logger.error(error_message, exc_info=True)  # 记录详细错误堆栈
        print(error_message)
        return error_message

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            user_input = " ".join(sys.argv[1:])
        else:
            user_input = input("你: ")
        
        logger.info("程序启动")
        logger.info(f"用户输入模式: {'命令行参数' if len(sys.argv) > 1 else '交互输入'}")
        
        print("\n智谱AI: ", end="")
        result = chat_with_zhipuai(user_input)
        logger.info("对话完成")
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        print("\n\n程序被用户中断")
    except Exception as e:
        error_msg = f"程序运行出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\n{error_msg}")