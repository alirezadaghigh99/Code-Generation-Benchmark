            async def infer_from_workflow(
                workflow_request: WorkflowSpecificationInferenceRequest,
                background_tasks: BackgroundTasks,
            ) -> WorkflowInferenceResponse:
                return await process_workflow_inference_request(
                    workflow_request=workflow_request,
                    workflow_specification=workflow_request.specification,
                    background_tasks=background_tasks if not LAMBDA else None,
                )