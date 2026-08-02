import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ApiService } from './api.service';
import { ComplianceReport, MessageResponse } from '../models';

export interface ReportGenerateRequest {
  job_id: string;
  report_type?: string;
}

export interface ReportFilters {
  study_id?: string;
  job_id?: string;
  offset?: number;
  limit?: number;
}

@Injectable({ providedIn: 'root' })
export class ReportsService extends ApiService {
  list(filters: ReportFilters = {}): Observable<ComplianceReport[]> {
    return this.get<ComplianceReport[]>('/reports', filters as Record<string, string | number | boolean | null | undefined>);
  }

  generate(body: ReportGenerateRequest): Observable<ComplianceReport> {
    return this.post<ComplianceReport>('/reports', body);
  }

  sign(id: string): Observable<ComplianceReport> {
    return this.post<ComplianceReport>(`/reports/${id}/sign`, {});
  }

  deleteReport(id: string): Observable<void> {
    return this.delete<void>(`/reports/${id}`);
  }

  getDownloadUrl(id: string): string {
    return `/api/v1/reports/${id}/download`;
  }

  downloadBlob(id: string): Observable<Blob> {
    return this.http.get(`${this.base}/reports/${id}/download`, { responseType: 'blob' });
  }
}
