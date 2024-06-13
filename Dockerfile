FROM rapidsai/rapidsai

COPY ./grid_search_cv_gpu.py ./workspace/grid_search_cv_gpu.py
COPY ./outputs2/cleaned_data_duplicate.csv ./workspace/cleaned_data_duplicate.csv

CMD ["python", "./workspace/grid_search_cv_gpu.py"]
